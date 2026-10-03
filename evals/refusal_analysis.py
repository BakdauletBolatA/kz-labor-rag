"""Отказ по скору реранкера: офлайн-разбор на сохранённом прогоне ответов.

    python evals/refusal_analysis.py                  # dev, последний прогон Ollama
    python evals/refusal_analysis.py --split test --tau 0.35   # проверка порога

Идея: решение «отказаться» принимает код по скору лучшего фрагмента, а не только
модель. Если лучший фрагмент набрал меньше порога, ответ заменяется отказом.

Ответы при температуре 0 детерминированы, поэтому замена ответа отказом не
требует новой генерации: правка применяется к записанному прогону. Отказ на
вопросе без ответа в кодексе судья считает верным, на вопросе с ответом —
неверным; те же правила применяет и ``answer_eval``.

Порог выбирается на dev; на test он только проверяется (``--tau``).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

from kz_labor_rag.cli import build_retriever
from kz_labor_rag.config import load_config, load_env_file
from kz_labor_rag.eval.citations import REFUSAL
from kz_labor_rag.eval.dataset import load_dataset
from kz_labor_rag.eval.experiments import cell_config

_spec = importlib.util.spec_from_file_location(
    "answer_eval", Path(__file__).resolve().parent / "answer_eval.py"
)
answer_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(answer_eval)

DEFAULT_CELL = answer_eval.DEFAULT_CELL


def apply_threshold(records: list[dict], scores: dict[str, float], tau: float) -> list[dict]:
    """Копия записей, где ответы с лучшим скором ниже порога заменены отказом."""
    out = copy.deepcopy(records)
    for r in out:
        if r.get("error") or scores.get(r["id"], 1.0) >= tau:
            continue
        r.update(
            answer=REFUSAL,
            refused=True,
            withheld=False,
            citation_hit=False,
            citations=[],
            invalid_citations=[],
        )
        r["judge"] = {
            "correctness": "correct" if r["unanswerable"] else "incorrect",
            "groundedness": "grounded",
        }
    return out


def sweep(
    records: list[dict], scores: dict[str, float], taus: list[float], seed: int
) -> list[dict]:
    rows = []
    for tau in taus:
        agg = answer_eval.aggregate(apply_threshold(records, scores, tau), seed=seed)
        answerable = [r for r in apply_threshold(records, scores, tau) if not r["unanswerable"]]
        unanswerable = [r for r in apply_threshold(records, scores, tau) if r["unanswerable"]]
        rows.append(
            {
                "tau": tau,
                "correctness": agg["correctness"],
                "ci": agg["ci"]["correctness"],
                "refused_answerable": answer_eval.mean(r["refused"] for r in answerable),
                "refused_unanswerable": answer_eval.mean(r["refused"] for r in unanswerable),
            }
        )
    return rows


def top_scores(config, questions, k: int) -> dict[str, float]:
    retriever = build_retriever(config)
    retriever.warmup()
    return {q.id: retriever.search(q.question, k)[0].score for q in questions}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--tau", type=float, default=None, help="проверить один порог")
    parser.add_argument("--generator", default=None)
    parser.add_argument("--results-dir", default="evals/results")
    args = parser.parse_args()
    load_env_file()

    base = load_config()
    base.data["eval"]["split"] = args.split
    seed, k = int(base.get("eval.seed")), int(base.get("eval.k"))
    dataset = load_dataset(base.path_of("eval.dataset"))
    questions = list(dataset.in_split(args.split))

    run_path = answer_eval.latest_run(
        Path(args.results_dir), args.generator or "ollama", args.split
    )
    records = json.loads(run_path.read_text("utf-8"))["questions"]
    config = cell_config(base, *DEFAULT_CELL)
    scores = top_scores(config, questions, k)

    if args.tau is not None:
        taus = [0.0, args.tau]
    else:
        values = sorted(scores.values())
        taus = [0.0] + [values[int(len(values) * f)] for f in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)]
    rows = sweep(records, scores, taus, seed)

    lines = [
        f"Refusal by top reranker score, {args.split} split ({len(records)} answers from "
        f"`{run_path.name}`). tau = 0 is the run as it was.",
        "",
        "| tau | correctness [95% CI] | answerable refused | unanswerable refused |",
        "|---|---|---|---|",
    ]
    for r in rows:
        low, high = r["ci"]
        lines.append(
            f"| {r['tau']:.3f} | {r['correctness']:.3f} [{low:.2f}, {high:.2f}] | "
            f"{r['refused_answerable']:.3f} | {r['refused_unanswerable']:.3f} |"
        )
    table = "\n".join(lines) + "\n"
    print(table)
    name = "refusal_analysis.md" if args.split == "dev" else f"refusal_analysis_{args.split}.md"
    Path(args.results_dir, name).write_text(
        f"<!-- generated by evals/refusal_analysis.py from {run_path.name} -->\n" + table, "utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
