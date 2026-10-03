"""Качество ответов: правильность, обоснованность, честные отказы.

    python evals/answer_eval.py                     # прогон по проверенным вопросам
    python evals/answer_eval.py --prepare-labels 20 # ответы для ручной разметки
    python evals/answer_eval.py --label             # разметить их (интерактивно)
    python evals/answer_eval.py --agreement         # согласие судьи с разметкой
    python evals/answer_eval.py --retry-failed      # переспросить упавшие вопросы
    python evals/answer_eval.py --rescore           # пересчитать таблицу без вызовов
    python evals/answer_eval.py --compare A.json B.json  # парное сравнение двух прогонов

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
from dataclasses import asdict
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
from kz_labor_rag.eval.citations import cite, is_refusal
from kz_labor_rag.eval.dataset import QUESTION_TYPES, load_dataset
from kz_labor_rag.eval.experiments import cell_config
from kz_labor_rag.eval.factory import build_forced_generator, build_generator
from kz_labor_rag.eval.judge import format_context
from kz_labor_rag.eval.stats import bootstrap_ci, paired_difference

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


def groundedness_of(answer_text: str, refused: bool, judged: str | None) -> str | None:
    """Обоснованность с учётом правила: отказ без утверждений обоснован.

    Решает код, а не судья: gpt-5 ставил отказу «ungrounded», если ответ в
    кодексе был, — это оценка правильности, не обоснованности. Исходная
    оценка судьи в результатах остаётся как есть.
    """
    if refused or is_refusal(answer_text):
        return "grounded"
    return judged


def mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def aggregate(records: list[dict], seed: int | None = None) -> dict:
    """Агрегаты прогона; с ``seed`` — ещё и 95% bootstrap-интервалы по вопросам."""
    # Упавшая генерация (ошибка Ollama, обрыв) — не «ответил плохо», а «не
    # измерено»: в метрики не идёт, но число таких ответов видно в таблице.
    failed = [r for r in records if r.get("error")]
    records = [r for r in records if not r.get("error")]
    answerable = [r for r in records if not r["unanswerable"]]
    unanswerable = [r for r in records if r["unanswerable"]]
    valid = sum(len(r["citations"]) for r in records)
    invalid = sum(len(r["invalid_citations"]) for r in records)
    judged = [r for r in records if r.get("judge", {}).get("correctness")]

    def correctness(r) -> float:
        return CORRECTNESS[r["judge"]["correctness"]]

    def groundedness(r) -> float:
        verdict = groundedness_of(r.get("answer", ""), r["refused"], r["judge"]["groundedness"])
        return GROUNDEDNESS[verdict]

    samples = {
        "answer_rate": [float(not r["refused"]) for r in answerable],
        "citation_hit": [float(r["citation_hit"]) for r in answerable],
        "correct_refusal": [float(r["refused"]) for r in unanswerable],
        "correctness": [correctness(r) for r in judged],
        "groundedness": [groundedness(r) for r in judged],
    }
    by_type: dict[str, dict] = {}
    for r in judged:
        by_type.setdefault(r.get("type", "?"), []).append(correctness(r))
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        **{name: mean(values) for name, values in samples.items()},
        "citation_validity": valid / (valid + invalid) if valid + invalid else None,
        "withheld": mean(r["withheld"] for r in records),
        "n_failed": len(failed),
        "n_judged": len(judged),
        "ci": (
            {name: bootstrap_ci(values, seed=seed) for name, values in samples.items()}
            if seed is not None
            else {}
        ),
        "by_type": {
            t: {
                "n": len(v),
                "correctness": mean(v),
                "ci": bootstrap_ci(v, seed=seed) if seed is not None else None,
            }
            for t, v in sorted(by_type.items())
        },
        "generation_p50_ms": (
            statistics.median(r["latency_ms"]["generation"] for r in records) if records else None
        ),
    }


def fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def with_ci(agg: dict, name: str) -> str:
    value, ci = agg.get(name), (agg.get("ci") or {}).get(name)
    if value is None:
        return "—"
    return f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{value:.3f}"


def render(cell: str, agg: dict, generator: dict, judge_note: str) -> str:
    failed = f" (+{n} failed to generate, not scored)" if (n := agg.get("n_failed")) else ""
    lines = [
        f"Cell: `{cell}`, generator: `{generator.get('model')}` "
        f"({generator.get('prompt')}). Verified questions: "
        f"{agg['n_answerable']} answerable, {agg['n_unanswerable']} unanswerable{failed}. "
        f"{judge_note} Intervals: 95% bootstrap over questions.",
        "",
        "| answer_rate | citation_hit | citation_validity | withheld | correct_refusal | "
        "correctness | groundedness | n_judged | generation p50, s |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {with_ci(agg, 'answer_rate')} | {with_ci(agg, 'citation_hit')} | "
        f"{fmt(agg['citation_validity'])} | {fmt(agg['withheld'])} | "
        f"{with_ci(agg, 'correct_refusal')} | {with_ci(agg, 'correctness')} | "
        f"{with_ci(agg, 'groundedness')} | {agg['n_judged']} | "
        f"{fmt(agg['generation_p50_ms'] and agg['generation_p50_ms'] / 1000)} |",
    ]
    if agg.get("by_type"):
        lines += [
            "",
            "Correctness by question type:",
            "",
            "| type | n | correctness |",
            "|---|---|---|",
        ]
        for t, v in agg["by_type"].items():
            ci = v.get("ci")
            value = f"{v['correctness']:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
            lines.append(f"| {t} | {v['n']} | {value} |")
    return "\n".join(lines) + "\n"


def table_name(stem: str, split: str) -> str:
    """Файл таблицы: для dev — прежнее имя, для остальных частей — с суффиксом."""
    return f"{stem}.md" if split == "dev" else f"{stem}_{split}.md"


def checkpoint_path(results_dir: Path, generator: str | None, split: str) -> Path:
    """Свой файл прогресса на каждый генератор и часть набора: параллельный прогон
    другой модели или другой части не должен начинать заново чужой незавершённый."""
    parts = ["partial", *([generator] if generator else []), *([split] if split != "dev" else [])]
    return results_dir / "answers" / ("-".join(parts) + ".jsonl")


def setup(cell, generator_name: str | None = None, split: str | None = None):
    base = load_config()
    if split:
        base.data["eval"]["split"] = split
    config = cell_config(base, *cell)
    config.data["generation"]["enabled"] = True
    if generator_name:
        # Облачная модель для сравнения: меняется только генератор, промпт тот же.
        alternative = base.get(f"generation_alternatives.{generator_name}")
        config.data["generation"].update(alternative, name=generator_name)
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
    base, config, code, retriever, generator = setup(args.cell, args.generator, args.split)
    split = base.get("eval.split")
    if generator.descriptor.get("backend") == "disabled":
        print(f"Генератор недоступен: {generator.descriptor.get('reason')}")
        return 1
    dataset = load_dataset(base.path_of("eval.dataset"))
    questions = list(dataset.in_split(split))
    judge, judge_reason = make_judge(base)
    k = int(base.get("eval.k"))

    dataset_sha = hashlib.sha256(Path(dataset.path).read_bytes()).hexdigest()
    checkpoint = Checkpoint(
        checkpoint_path(Path(args.results_dir), args.generator, split),
        key={
            "cell": "/".join(args.cell),
            "generator": generator.descriptor,
            "judge": judge.descriptor if judge else None,
            "dataset_sha256": dataset_sha,
            "split": split,
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

    agg = aggregate(records, seed=int(base.get("eval.seed")))
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
        "split": split,
        "aggregates": agg,
        "questions": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (out / table_name("answer_table", split)).write_text(
        f"<!-- generated by evals/answer_eval.py from {path.name} -->\n{table}", "utf-8"
    )
    print(f"Записано: {path}")
    checkpoint.done()
    return 0


def latest_run(out: Path, backend: str, split: str = "dev") -> Path:
    """Последний полный прогон этого генератора на этой части набора.

    Прогоны разных моделей и разных частей лежат рядом; результаты, записанные до
    появления частей, относятся к dev."""
    runs = []
    for p in sorted((out / "answers").glob("2*.json")):
        payload = json.loads(p.read_text("utf-8"))
        if (
            payload.get("generator", {}).get("backend") == backend
            and payload.get("split", "dev") == split
            and not payload.get("variant")
        ):
            runs.append(p)
    if not runs:
        raise SystemExit(f"Нет прогонов генератора {backend} ({split}) в {out / 'answers'}")
    return runs[-1]


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
    backend = args.generator or "ollama"
    split = args.split or "dev"
    latest = latest_run(out, backend, split)
    payload = json.loads(latest.read_text("utf-8"))
    base, config, code, retriever, generator = setup(
        payload["cell"].split("/"), args.generator, split
    )
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

    agg = aggregate(payload["questions"], seed=int(base.get("eval.seed")))
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
    (out / table_name("answer_table", payload.get("split", "dev"))).write_text(
        f"<!-- generated by evals/answer_eval.py from {path.name} -->\n{table}", "utf-8"
    )
    print("\n" + table + f"\nЗаписано: {path} (повторено: {', '.join(retry)})")
    return 0


def cmd_rescore(args) -> int:
    """Пересчитать агрегаты последнего прогона из сохранённых ответов и оценок.

    Нужен, когда меняется правило подсчёта, а не ответы: без генерации и без
    обращений к судье.
    """
    out = Path(args.results_dir)
    # Явный путь — когда в таблицу должен попасть не последний прогон, а тот,
    # на котором работает сервис (например, после отката неудачной итерации).
    latest = (
        Path(args.rescore)
        if args.rescore != "latest"
        else latest_run(out, args.generator or "ollama", args.split or "dev")
    )
    payload = json.loads(latest.read_text("utf-8"))
    agg = aggregate(payload["questions"], seed=int(load_config().get("eval.seed")))
    payload["aggregates"] = agg
    payload["rescored_from"] = latest.name
    judge = payload.get("judge") or {}
    note = (
        f"Judge: `{judge['model']}` ({judge['prompt']})."
        if judge.get("model")
        else "Judge not run: no API key for the configured provider."
    )
    table = render(payload["cell"], agg, payload["generator"], note)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / "answers" / f"{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (out / table_name("answer_table", payload.get("split", "dev"))).write_text(
        f"<!-- generated by evals/answer_eval.py from {path.name} -->\n{table}", "utf-8"
    )
    print("\n" + table + f"\nЗаписано: {path} (пересчитано из {latest.name})")
    return 0


def compare_runs(before: list[dict], after: list[dict], *, seed: int) -> list[dict]:
    """Парные разницы «после минус до» по вопросам, которые есть в обоих прогонах.

    correct_refusal считается отдельно по вопросам без ответа: изменение,
    которое снижает число ложных отказов, легко заодно ломает честные.
    """
    a = {r["id"]: r for r in before if not r.get("error")}
    b = {r["id"]: r for r in after if not r.get("error")}
    ids = sorted(set(a) & set(b))
    judged = [
        i
        for i in ids
        if (a[i].get("judge") or {}).get("correctness")
        and (b[i].get("judge") or {}).get("correctness")
    ]

    def ground(r) -> float:
        verdict = groundedness_of(r.get("answer", ""), r["refused"], r["judge"]["groundedness"])
        return GROUNDEDNESS[verdict]

    metrics = {
        "correctness (answerable)": (
            [i for i in judged if not a[i]["unanswerable"]],
            lambda r: CORRECTNESS[r["judge"]["correctness"]],
        ),
        "groundedness": (judged, ground),
        "answer_rate": (
            [i for i in ids if not a[i]["unanswerable"]],
            lambda r: float(not r["refused"]),
        ),
        "citation_hit": (
            [i for i in ids if not a[i]["unanswerable"]],
            lambda r: float(r["citation_hit"]),
        ),
        "correct_refusal": (
            [i for i in ids if a[i]["unanswerable"]],
            lambda r: float(r["refused"]),
        ),
    }
    rows = []
    for name, (subset, score) in metrics.items():
        if not subset:
            continue
        diff = paired_difference(
            [score(a[i]) for i in subset], [score(b[i]) for i in subset], seed=seed
        )
        rows.append(
            {
                "name": name,
                "n": len(subset),
                "before": mean(score(a[i]) for i in subset),
                "after": mean(score(b[i]) for i in subset),
                **asdict(diff),
                "significant": diff.significant,
            }
        )
    return rows


def cmd_compare(args) -> int:
    before_path, after_path = (Path(p) for p in args.compare)
    before = json.loads(before_path.read_text("utf-8"))
    after = json.loads(after_path.read_text("utf-8"))
    rows = compare_runs(
        before["questions"], after["questions"], seed=int(load_config().get("eval.seed"))
    )
    label = lambda run: (  # noqa: E731
        f"`{run['generator'].get('model')}` with `{run['generator'].get('prompt')}` ({run['cell']})"
    )
    lines = [
        f"Paired comparison on the same questions: {label(before)} → {label(after)}. "
        "Δ is after minus before, 95% paired bootstrap; `*` — the interval excludes zero.",
        "",
        "| metric | n | before | after | Δ [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mark = " *" if r["significant"] else ""
        lines.append(
            f"| {r['name']} | {r['n']} | {r['before']:.3f} | {r['after']:.3f} | "
            f"{r['mean']:+.3f} [{r['low']:+.2f}, {r['high']:+.2f}]{mark} |"
        )
    table = "\n".join(lines) + "\n"
    print(table)
    Path(args.results_dir, args.compare_out).write_text(
        "<!-- generated by evals/answer_eval.py --compare "
        f"{before_path.name} {after_path.name} -->\n" + table,
        "utf-8",
    )
    return 0


def refused_above(records: list[dict], scores: dict[str, float], tau: float) -> list[str]:
    """Отказы, у которых лучший фрагмент набрал не ниже порога.

    Метка «есть ли ответ в кодексе» правилу неизвестна — как и в работающей
    системе: решение принимается только по скору."""
    return [
        r["id"]
        for r in records
        if not r.get("error") and r["refused"] and scores.get(r["id"], 0.0) >= tau
    ]


def cmd_apply_force(args) -> int:
    """Применить правило ForceAnswerGenerator к сохранённому прогону.

    Ответы при температуре 0 детерминированы, поэтому заново спрашиваются только
    отказы выше порога; остальные записи переносятся как есть. Результат —
    прогон-вариант (``variant``), последним «прогоном сервиса» он не считается.
    """
    tau = float(args.apply_force)
    out = Path(args.results_dir)
    split = args.split or "dev"
    latest = latest_run(out, args.generator or "ollama", split)
    payload = json.loads(latest.read_text("utf-8"))
    base, config, code, retriever, generator = setup(
        payload["cell"].split("/"), args.generator, split
    )
    forced = build_forced_generator(config)
    judge, _ = make_judge(base)
    k = int(base.get("eval.k"))
    questions = {q.id: q for q in load_dataset(base.path_of("eval.dataset")).in_split(split)}

    refused = [r for r in payload["questions"] if not r.get("error") and r["refused"]]
    scores = {r["id"]: retriever.search(questions[r["id"]].question, k)[0].score for r in refused}
    targets = refused_above(payload["questions"], scores, tau)
    changed = []
    for i, record in enumerate(payload["questions"]):
        if record["id"] not in targets:
            continue
        log.info("Повтор без отказа: %s (скор %.3f)", record["id"], scores[record["id"]])
        new = answer(questions[record["id"]], retriever, forced, k)
        if new["error"] or new["refused"]:
            record["force_tried"] = True
            continue
        if judge is not None:
            run_judge(judge, questions[record["id"]], code, new)
        new.update(forced=True, answer_before=record["answer"], top_score=scores[record["id"]])
        payload["questions"][i] = {key: v for key, v in new.items() if key != "_hits"}
        changed.append(record["id"])

    payload["variant"] = f"force@{tau}"
    payload["force_answer_above"] = tau
    payload["forced_from"] = latest.name
    payload["generator"] = {
        **payload["generator"],
        "force_answer_above": str(tau),
        "forced_prompt": forced.descriptor.get("prompt", ""),
    }
    payload["aggregates"] = aggregate(payload["questions"], seed=int(base.get("eval.seed")))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / "answers" / f"{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(
        f"Порог {tau}: повторно спрошено {len(targets)}, отказ заменён ответом у {len(changed)}"
        f" ({', '.join(changed) or '—'}).\nЗаписано: {path}\n"
        f"Сравнить: --compare {latest} {path}"
    )
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
        machine = [
            groundedness_of(r["answer"], False, r["judge"][dim])
            if dim == "groundedness"
            else r["judge"][dim]
            for r in pairs
        ]
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
    parser.add_argument(
        "--split",
        default=None,
        choices=["dev", "test", "all"],
        help="часть набора (по умолчанию — eval.split из конфига: dev)",
    )
    parser.add_argument(
        "--compare-out",
        default="answer_comparison.md",
        help="куда записать таблицу --compare (в каталоге результатов)",
    )
    parser.add_argument(
        "--generator",
        default=None,
        metavar="NAME",
        help="облачная модель из generation_alternatives вместо локальной",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-labels", type=int, metavar="N")
    mode.add_argument("--label", action="store_true")
    mode.add_argument("--agreement", action="store_true")
    mode.add_argument("--retry-failed", action="store_true")
    mode.add_argument(
        "--apply-force",
        type=float,
        metavar="TAU",
        help="повторить отказы со скором лучшего фрагмента не ниже TAU промптом без отказа",
    )
    mode.add_argument(
        "--rescore",
        nargs="?",
        const="latest",
        metavar="RUN.json",
        help="пересчитать таблицу из сохранённого прогона (по умолчанию — последнего)",
    )
    mode.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"))
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
    if args.apply_force is not None:
        return cmd_apply_force(args)
    if args.rescore:
        return cmd_rescore(args)
    if args.compare:
        return cmd_compare(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
