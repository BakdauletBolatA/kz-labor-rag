"""Качество ответов: правильность, обоснованность, честные отказы.

    python evals/answer_eval.py                     # прогон по проверенным вопросам
    python evals/answer_eval.py --prepare-labels 20 # ответы для ручной разметки
    python evals/answer_eval.py --label             # разметить их (интерактивно)
    python evals/answer_eval.py --agreement         # согласие судьи с разметкой
    python evals/answer_eval.py --retry-failed      # переспросить упавшие вопросы

Поиск и генерация берутся из одной ячейки сравнительной таблицы (по умолчанию
нарезка по пунктам с заголовком статьи, гибридный поиск и реранкинг —
то же, на чём работает сервис), генерация — локальная
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

DEFAULT_CELL = ("clause+header", "e5-base", "hybrid+rerank")
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
    # Упавшая генерация (ошибка Ollama, обрыв) — не «ответил плохо», а «не
    # измерено»: в метрики не идёт, но число таких ответов видно в таблице.
    failed = [r for r in records if r.get("error")]
    records = [r for r in records if not r.get("error")]
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
        "n_failed": len(failed),
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
        f"{agg['n_answerable']} answerable, {agg['n_unanswerable']} unanswerable"
        f"{f' (+{n} failed to generate, not scored)' if (n := agg.get('n_failed')) else ''}. "
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


class Checkpoint:
    """Ответы, готовые до обрыва прогона.

    Каждый ответ дописывается в файл сразу. Первая строка — ключ настроек:
    если ячейка, модель, промпт или набор изменились, старые ответы к новому
    прогону не подмешиваются и файл начинается заново.
    """

    def __init__(self, path: Path, key: dict) -> None:
        self.path = path
        self.key = key
        self.records: list[dict] = []
        if path.exists():
            lines = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
            if lines and lines[0] == {"key": key}:
                self.records = lines[1:]
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key}, ensure_ascii=False) + "\n", "utf-8")

    @property
    def done_ids(self) -> set[str]:
        return {r["id"] for r in self.records}

    def add(self, record: dict) -> None:
        clean = {k: v for k, v in record.items() if k != "_hits"}
        self.records.append(clean)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    def done(self) -> None:
        self.path.unlink(missing_ok=True)


def cmd_run(args) -> int:
    base, config, code, retriever, generator = setup(args.cell)
    dataset = load_dataset(base.path_of("eval.dataset"))
    questions = list(dataset.verified)
    judge, judge_reason = make_judge(base)
    k = int(base.get("eval.k"))

    dataset_sha = hashlib.sha256(Path(dataset.path).read_bytes()).hexdigest()
    checkpoint = Checkpoint(
        Path(args.results_dir) / "answers" / "partial.jsonl",
        key={
            "cell": "/".join(args.cell),
            "generator": generator.descriptor,
            "judge": judge.descriptor if judge else None,
            "dataset_sha256": dataset_sha,
        },
    )
    if checkpoint.records:
        log.info("Продолжаю прерванный прогон: готово %d ответов", len(checkpoint.records))
    for q in questions:
        if q.id in checkpoint.done_ids:
            continue
        log.info("Вопрос %s", q.id)
        record = answer(q, retriever, generator, k)
        if judge is not None and not record["error"]:
            run_judge(judge, q, code, record)
        checkpoint.add(record)
    records = checkpoint.records

    agg = aggregate(records)
    cell = "/".join(args.cell)
    note = (
        f"Judge: `{judge.model}` ({judge.descriptor['prompt']})."
        if judge
        else f"Judge not run: no API key for `answer_judge.provider` = "
        f"{base.get('answer_judge.provider')}."
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
        "dataset_sha256": dataset_sha,
        "aggregates": agg,
        "questions": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (out / "answer_table.md").write_text(
        f"<!-- generated by evals/answer_eval.py from {path.name} -->\n{table}", "utf-8"
    )
    print(f"Записано: {path}")
    checkpoint.done()
    return 0


def needs_retry(records: list[dict], *, judge_on: bool) -> list[str]:
    """Вопросы, где упала генерация или (если судья есть) нет его оценки."""
    return [
        r["id"]
        for r in records
        if r.get("error") or (judge_on and not (r.get("judge") or {}).get("correctness"))
    ]


def cmd_retry_failed(args) -> int:
    """Переспросить только упавшие вопросы последнего прогона и пересчитать таблицу."""
    out = Path(args.results_dir)
    latest = sorted((out / "answers").glob("2*.json"))[-1]
    payload = json.loads(latest.read_text("utf-8"))
    base, config, code, retriever, generator = setup(payload["cell"].split("/"))
    judge, _ = make_judge(base)
    retry = needs_retry(payload["questions"], judge_on=judge is not None)
    if not retry:
        print(f"В {latest.name} переспрашивать нечего.")
        return 0
    questions = {q.id: q for q in load_dataset(base.path_of("eval.dataset"))}
    k = int(base.get("eval.k"))
    for i, record in enumerate(payload["questions"]):
        if record["id"] not in retry:
            continue
        log.info("Повтор %s", record["id"])
        new = answer(questions[record["id"]], retriever, generator, k)
        if judge is not None and not new["error"]:
            run_judge(judge, questions[record["id"]], code, new)
        payload["questions"][i] = {key: v for key, v in new.items() if key != "_hits"}

    agg = aggregate(payload["questions"])
    payload["aggregates"] = agg
    payload["retried"] = retry
    note = (
        f"Judge: `{judge.model}` ({judge.descriptor['prompt']})."
        if judge
        else "Judge not run: no API key for the configured provider."
    )
    table = render(payload["cell"], agg, generator.descriptor, note)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / "answers" / f"{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (out / "answer_table.md").write_text(
        f"<!-- generated by evals/answer_eval.py from {path.name} -->\n{table}", "utf-8"
    )
    print("\n" + table + f"\nЗаписано: {path} (повторено: {', '.join(retry)})")
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


def set_label(row_id: str, correctness: str | None, groundedness: str | None) -> None:
    """Записать одну метку, перечитав файл: другая открытая сессия могла уже что-то сохранить."""
    rows = read_labels()
    for row in rows:
        if row["id"] == row_id:
            row["label_correctness"], row["label_groundedness"] = correctness, groundedness
    write_labels(rows)


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
        set_label(row["id"], correctness, grounded)
    rows = read_labels()
    done = sum(1 for r in rows if r["label_correctness"] and r["label_groundedness"])
    print(f"\nРазмечено {done} из {len(rows)}.")
    return 0


def agreement_table(labelled: list[dict], model: str) -> str | None:
    """Согласие судьи с ручной разметкой; None, если судья не оценил ни одной пары."""
    pairs = [r for r in labelled if (r.get("judge") or {}).get("correctness")]
    if not pairs:
        return None
    lines = [
        f"Judge `{model}` vs manual labels on {len(pairs)} answers "
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
    return "\n".join(lines) + "\n"


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

    table = agreement_table(labelled, judge.model)
    if table is None:
        print(
            "Ни один размеченный ответ не получил оценку судьи (ошибки в поле judge "
            f"в {LABELS}). Запустите --agreement ещё раз."
        )
        return 1
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
    mode.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.prepare_labels:
        return cmd_prepare_labels(args)
    if args.label:
        return cmd_label(args)
    if args.agreement:
        return cmd_agreement(args)
    if args.retry_failed:
        return cmd_retry_failed(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
