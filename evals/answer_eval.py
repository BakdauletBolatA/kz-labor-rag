"""Качество ответов: правильность, обоснованность, честные отказы.

    python evals/answer_eval.py                     # прогон по проверенным вопросам
    python evals/answer_eval.py --prepare-labels 20 # ответы для ручной разметки
    python evals/answer_eval.py --label             # разметить их (интерактивно)
    python evals/answer_eval.py --agreement         # согласие судьи с разметкой

Поиск и генерация берутся из одной ячейки сравнительной таблицы (по умолчанию
нарезка по пунктам с заголовком статьи и гибридный поиск), генерация — локальная
модель из ``generation`` конфига.

Детерминированные метрики (без LLM):
- answer_rate — доля вопросов с ответом в кодексе, на которые система ответила;
- citation_hit — доля таких вопросов, где хотя бы одна ссылка ответа попала в
  эталонные пункты;
- citation_validity — доля ссылок модели на пункты, которые она видела;
- withheld — доля ответов, скрытых из-за отсутствия верных ссылок;
- correct_refusal — доля вопросов без ответа в кодексе, на которые система
  честно отказалась отвечать.

LLM-судья (секция ``answer_judge``: OpenAI или Anthropic, нужен ключ вендора)
оценивает correctness по эталонным пунктам и groundedness по показанным
фрагментам. Без ключа эти метрики записываются как null с причиной.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from kz_labor_rag.cli import build_retriever
from kz_labor_rag.config import load_config, load_env_file
from kz_labor_rag.corpus.parser import parse_file
from kz_labor_rag.eval.answer_judge import (
    CORRECTNESS,
    GROUNDEDNESS,
    build_answer_judge,
    cohen_kappa,
    reference_for,
)
from kz_labor_rag.eval.citations import cite
from kz_labor_rag.eval.dataset import QUESTION_TYPES, load_dataset
from kz_labor_rag.eval.experiments import cell_config
from kz_labor_rag.eval.factory import build_generator
from kz_labor_rag.eval.judge import format_context

log = logging.getLogger("answer_eval")

DEFAULT_CELL = ("clause+header", "e5-base", "hybrid")
LABELS = Path("evals/manual_labels.jsonl")


def answer(question, retriever, generator, k: int) -> dict:
    t0 = time.perf_counter()
    hits = list(retriever.search(question.question, k))
    t1 = time.perf_counter()
    generation = generator.generate(question.question, hits)
    t2 = time.perf_counter()
    required = [cite(r.article, r.clause) for r in question.required_clauses]
    return {
        "id": question.id,
        "type": question.type,
        "origin": question.origin,
        "question": question.question,
        "unanswerable": question.is_unanswerable,
        "required_clauses": required,
        "retrieved": [
            {"chunk_id": h.chunk_id, "clauses": [cite(a, c) for a, c in h.chunk.spans]}
            for h in hits
        ],
        "answer": generation.answer,
        "raw": generation.raw,
        "citations": list(generation.citations),
        "invalid_citations": list(generation.invalid_citations),
        "citation_hit": bool(set(generation.citations) & set(required)),
        "refused": generation.refused,
        "withheld": generation.withheld,
        "error": generation.error,
        "tokens": {"input": generation.input_tokens, "output": generation.output_tokens},
        "latency_ms": {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000},
        "_hits": hits,
    }


def mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def aggregate(records: list[dict]) -> dict:
    answerable = [r for r in records if not r["unanswerable"]]
    unanswerable = [r for r in records if r["unanswerable"]]
    valid = sum(len(r["citations"]) for r in records)
    invalid = sum(len(r["invalid_citations"]) for r in records)
    judged = [r for r in records if r.get("judge", {}).get("correctness")]
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answer_rate": mean(not r["refused"] for r in answerable),
        "citation_hit": mean(r["citation_hit"] for r in answerable),
        "citation_validity": valid / (valid + invalid) if valid + invalid else None,
        "withheld": mean(r["withheld"] for r in records),
        "correct_refusal": mean(r["refused"] for r in unanswerable),
        "n_judged": len(judged),
        "correctness": mean(CORRECTNESS[r["judge"]["correctness"]] for r in judged),
        "groundedness": mean(GROUNDEDNESS[r["judge"]["groundedness"]] for r in judged),
        "generation_p50_ms": (
            statistics.median(r["latency_ms"]["generation"] for r in records) if records else None
        ),
    }


def fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def render(cell: str, agg: dict, generator: dict, judge_note: str) -> str:
    return (
        f"Cell: `{cell}`, generator: `{generator.get('model')}` "
        f"({generator.get('prompt')}). Verified questions: "
        f"{agg['n_answerable']} answerable, {agg['n_unanswerable']} unanswerable. "
        f"{judge_note}\n\n"
        "| answer_rate | citation_hit | citation_validity | withheld | correct_refusal | "
        "correctness | groundedness | n_judged | generation p50, s |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        f"| {fmt(agg['answer_rate'])} | {fmt(agg['citation_hit'])} | "
        f"{fmt(agg['citation_validity'])} | {fmt(agg['withheld'])} | "
        f"{fmt(agg['correct_refusal'])} | {fmt(agg['correctness'])} | "
        f"{fmt(agg['groundedness'])} | {agg['n_judged']} | "
        f"{fmt(agg['generation_p50_ms'] and agg['generation_p50_ms'] / 1000)} |\n"
    )


def setup(cell):
    base = load_config()
    config = cell_config(base, *cell)
    config.data["generation"]["enabled"] = True
    code = parse_file(base.path_of("corpus.raw_html"))
    retriever = build_retriever(config)
    retriever.warmup()
    return base, config, code, retriever, build_generator(config)


def make_judge(base):
    return build_answer_judge(base)


def run_judge(judge, question, code, record) -> None:
    verdict = judge.judge(
        question.question, reference_for(question, code), record["_hits"], record["answer"]
    )
    record["judge"] = {
        "correctness": verdict.correctness,
        "groundedness": verdict.groundedness,
        "reasoning": verdict.reasoning,
        "error": verdict.error,
    }


def cmd_run(args) -> int:
    base, config, code, retriever, generator = setup(args.cell)
    dataset = load_dataset(base.path_of("eval.dataset"))
    questions = list(dataset.verified)
    judge, judge_reason = make_judge(base)
    k = int(base.get("eval.k"))

    records = []
    for q in questions:
        log.info("Вопрос %s", q.id)
        record = answer(q, retriever, generator, k)
        if judge is not None and not record["error"]:
            run_judge(judge, q, code, record)
        records.append(record)

    agg = aggregate(records)
    cell = "/".join(args.cell)
    note = (
        f"Judge: `{judge.model}` ({judge.descriptor['prompt']})."
        if judge
        else f"Judge not run: {judge_reason}."
    )
    table = render(cell, agg, generator.descriptor, note)
    print("\n" + table)

    out = Path(args.results_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / "answers").mkdir(parents=True, exist_ok=True)
    path = out / "answers" / f"{stamp}.json"
    payload = {
        "cell": cell,
        "config": config.data,
        "generator": generator.descriptor,
        "judge": judge.descriptor if judge else {"skipped": judge_reason},
        "dataset_sha256": hashlib.sha256(Path(dataset.path).read_bytes()).hexdigest(),
        "aggregates": agg,
        "questions": [{k: v for k, v in r.items() if k != "_hits"} for r in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (out / "answer_table.md").write_text(
        f"<!-- generated by evals/answer_eval.py from {path.name} -->\n{table}", "utf-8"
    )
    print(f"Записано: {path}")
    return 0


def cmd_prepare_labels(args) -> int:
    if LABELS.exists():
        print(f"{LABELS} уже есть — в нём может быть ваша разметка, не перезаписываю.")
        return 1
    base, _, code, retriever, generator = setup(args.cell)
    dataset = load_dataset(base.path_of("eval.dataset"))
    rng = random.Random(int(base.get("eval.seed")))
    per_type = max(1, args.prepare_labels // len(QUESTION_TYPES))
    sample = []
    for qtype in QUESTION_TYPES:
        pool = sorted((q for q in dataset.ready if q.type == qtype), key=lambda q: q.id)
        sample += rng.sample(pool, min(per_type, len(pool)))

    rows = []
    for q in sample:
        log.info("Вопрос %s", q.id)
        record = answer(q, retriever, generator, int(base.get("eval.k")))
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "question": q.question,
                "reference": reference_for(q, code),
                "context": format_context(record["_hits"]),
                "answer": record["answer"],
                "raw": record["raw"],
                "citations": record["citations"],
                "label_correctness": None,
                "label_groundedness": None,
                "comment": "",
            }
        )
    LABELS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    print(f"Записано {len(rows)} ответов в {LABELS}. Разметка: python evals/answer_eval.py --label")
    return 0


def read_labels() -> list[dict]:
    return [json.loads(line) for line in LABELS.read_text("utf-8").splitlines() if line.strip()]


def write_labels(rows: list[dict]) -> None:
    tmp = LABELS.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    tmp.replace(LABELS)


def ask_choice(prompt: str, options: dict[str, str]) -> str | None:
    keys = "/".join(options)
    while True:
        raw = input(f"{prompt} [{keys}, s — пропустить]: ").strip().lower()
        if raw == "s":
            return None
        if raw in options:
            return options[raw]


def cmd_label(args) -> int:
    rows = read_labels()
    for i, row in enumerate(rows):
        if row["label_correctness"] and row["label_groundedness"]:
            continue
        print(f"\n{'=' * 78}\n[{i + 1}/{len(rows)}] {row['id']} ({row['type']})")
        print(f"Вопрос: {row['question']}\n\nЭталон:\n{row['reference']}")
        print(f"\nФрагменты, которые видела модель:\n{row['context']}")
        print(f"\nОтвет:\n{row['answer']}\n")
        correctness = ask_choice(
            "Правильность: c — верно, p — частично, i — неверно",
            {"c": "correct", "p": "partially_correct", "i": "incorrect"},
        )
        grounded = ask_choice(
            "Обоснованность фрагментами: g — да, p — частично, u — нет",
            {"g": "grounded", "p": "partially_grounded", "u": "ungrounded"},
        )
        row["label_correctness"], row["label_groundedness"] = correctness, grounded
        write_labels(rows)
    done = sum(1 for r in rows if r["label_correctness"] and r["label_groundedness"])
    print(f"\nРазмечено {done} из {len(rows)}.")
    return 0


def cmd_agreement(args) -> int:
    base = load_config()
    judge, reason = make_judge(base)
    if judge is None:
        print(f"Судья недоступен: {reason}")
        return 1
    rows = read_labels()
    labelled = [r for r in rows if r["label_correctness"] and r["label_groundedness"]]
    if not labelled:
        print("Нет размеченных ответов: python evals/answer_eval.py --label")
        return 1
    for row in labelled:
        if row.get("judge", {}).get("correctness"):
            continue
        # Судья видит те же фрагменты, что модель: они сохранены текстом.
        verdict = judge.judge(row["question"], row["reference"], row["context"], row["answer"])
        row["judge"] = {
            "correctness": verdict.correctness,
            "groundedness": verdict.groundedness,
            "reasoning": verdict.reasoning,
            "error": verdict.error,
        }
        write_labels(rows)

    pairs = [r for r in labelled if r.get("judge", {}).get("correctness")]
    lines = [
        f"Judge `{judge.model}` vs manual labels on {len(pairs)} answers "
        f"(generated by `evals/answer_eval.py --prepare-labels`).",
        "",
        "| dimension | raw agreement | Cohen's kappa |",
        "|---|---|---|",
    ]
    for dim in ("correctness", "groundedness"):
        human = [r[f"label_{dim}"] for r in pairs]
        machine = [r["judge"][dim] for r in pairs]
        raw = sum(h == m for h, m in zip(human, machine, strict=True)) / len(pairs)
        lines.append(f"| {dim} | {raw:.3f} | {fmt(cohen_kappa(human, machine))} |")
    table = "\n".join(lines) + "\n"
    print(table)
    Path(args.results_dir, "judge_agreement.md").write_text(
        "<!-- generated by evals/answer_eval.py --agreement -->\n" + table, "utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--cell", nargs=3, default=list(DEFAULT_CELL), metavar=("CHUNKING", "EMBEDDINGS", "METHOD")
    )
    parser.add_argument("--results-dir", default="evals/results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-labels", type=int, metavar="N")
    mode.add_argument("--label", action="store_true")
    mode.add_argument("--agreement", action="store_true")
    args = parser.parse_args()
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.prepare_labels:
        return cmd_prepare_labels(args)
    if args.label:
        return cmd_label(args)
    if args.agreement:
        return cmd_agreement(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
