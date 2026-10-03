"""Таблица «чанкинг × эмбеддинги × метод поиска».

    python evals/retrieval_eval.py                 # все ячейки
    python evals/retrieval_eval.py --only dense    # ячейки, где метод содержит строку

Для каждой ячейки строится (или переиспользуется) свой индекс в отдельной
таблице pgvector, прогоняются проверенные вопросы из ``eval.dataset`` и
считаются recall@5 и MRR по пунктам, статейный recall и задержка каждого
шага поиска. Метрики считает тот же ``EvalRunner``, что и ``kzrag-eval run``.

Результат: ``evals/results/retrieval/<время>.json`` со всеми ячейками и
выдачей по каждому вопросу, и ``evals/results/retrieval_table.md`` — таблица
последнего прогона.

Модели эмбеддингов сравниваются на одинаковых чанках: нарезку задаёт базовая
модель (e5), а скрипт проверяет, что тексты чанков в таблицах разных моделей
совпадают. Иначе разница в recall смешала бы модель с нарезкой.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kz_labor_rag.cli import build_retriever
from kz_labor_rag.config import load_config, load_env_file
from kz_labor_rag.corpus.chunker import build_tokenizer
from kz_labor_rag.eval.dataset import load_dataset
from kz_labor_rag.eval.experiments import CHUNKINGS, cell_config
from kz_labor_rag.eval.runner import EvalRunner
from kz_labor_rag.eval.stats import bootstrap_ci, paired_difference
from kz_labor_rag.retrieval.factory import build_store

log = logging.getLogger("retrieval_eval")

# Основная сетка: каждая нарезка × четыре метода на базовой модели. Сверху —
# сравнения этапа 3 на baseline-нарезке: второй анализатор BM25, реранкинг без
# гибрида и вторая модель эмбеддингов (только dense: на остальное модель не влияет).
GRID_METHODS = ("dense", "bm25-lemma", "hybrid", "hybrid+rerank")
CELLS: list[tuple[str, str, str]] = [(c, "e5-base", m) for c in CHUNKINGS for m in GRID_METHODS] + [
    ("fixed512", "e5-base", "bm25-stem"),
    ("fixed512", "e5-base", "dense+rerank"),
    ("fixed512", "bge-m3", "dense"),
    # Итерация «поиск»: вопросы, где поиск не находит ни одного нужного пункта.
    ("clause+header", "e5-base", "dense+rerank"),
    ("clause+header", "e5-base", "dense+rerank-k40"),
    ("clause+header", "e5-base", "hybrid+rerank-k40"),
    ("article", "e5-base", "dense+rerank"),
]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(result: dict, k: int, token_lengths: dict[str, int] | None = None) -> dict:
    """Агрегаты ячейки. ``token_lengths`` — длина каждого чанка в токенах.

    Размер выданного контекста считается рядом с recall: длинный чанк покрывает
    больше пунктов, и без этой колонки нарезка по статьям «выигрывала» бы
    просто тем, что отдаёт генератору больше текста.
    """
    agg = result["aggregates"]["primary"]
    latencies: dict[str, list[float]] = {}
    context: list[int] = []
    for question in result["questions"]:
        for step, ms in question["latency_ms"].items():
            if step != "generation":
                latencies.setdefault(step, []).append(ms)
        if token_lengths is not None:
            context.append(sum(token_lengths[c["chunk_id"]] for c in question["retrieved"][:k]))
    return {
        "n": agg.get("n", 0),
        f"recall@{k}": agg.get(f"recall@{k}"),
        f"strict_hit@{k}": agg.get(f"strict_hit@{k}"),
        "mrr": agg.get("mrr"),
        f"article_recall@{k}": agg.get(f"article_recall@{k}"),
        "latency_ms": {
            step: {"p50": percentile(v, 50), "p95": percentile(v, 95)}
            for step, v in latencies.items()
        },
        "chunks": result["index"].get("chunks"),
        "context_tokens": statistics.fmean(context) if context else None,
        "chunking_signature": result["index"].get("chunking_signature"),
        "failures": agg.get("retrieval_failures", []),
    }


BASELINE_CELL = "fixed512/e5-base/dense"


def add_intervals(rows: list[dict], runs: dict[str, dict], *, seed: int) -> None:
    """95% интервалы recall и MRR и парная разница recall с baseline.

    Разница считается парным bootstrap по одним и тем же вопросам: так из неё
    вычитается трудность вопроса, общая для обеих конфигураций.
    """

    def per_question(version: str, metric: str) -> dict[str, float]:
        return {q["id"]: q["metrics"][metric] for q in runs[version]["questions"]}

    base = per_question(BASELINE_CELL, "recall_at_k") if BASELINE_CELL in runs else None
    for row in rows:
        version = f"{row['chunking']}/{row['embeddings']}/{row['method']}"
        recall = per_question(version, "recall_at_k")
        rr = per_question(version, "reciprocal_rank")
        row["recall_ci"] = bootstrap_ci(list(recall.values()), seed=seed)
        row["mrr_ci"] = bootstrap_ci(list(rr.values()), seed=seed)
        if version == BASELINE_CELL:
            row["vs_baseline"] = "baseline"
        elif base is None:
            row["vs_baseline"] = None
        else:
            ids = sorted(set(base) & set(recall))
            diff = paired_difference([base[i] for i in ids], [recall[i] for i in ids], seed=seed)
            row["vs_baseline"] = {**asdict(diff), "significant": diff.significant}


def fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def with_ci(value, ci) -> str:
    if value is None:
        return "—"
    return fmt(value) if not ci else f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def fmt_delta(diff) -> str:
    if diff is None:
        return "—"
    if diff == "baseline":
        return "baseline"
    mark = " *" if diff["significant"] else ""
    return f"{diff['mean']:+.3f} [{diff['low']:+.2f}, {diff['high']:+.2f}]{mark}"


def render(rows: list[dict], k: int, n_info: dict) -> str:
    steps = ["dense", "bm25", "fusion", "rerank"]
    header = (
        f"| chunking | embeddings | method | chunks | context tokens@{k} | "
        f"recall@{k} [95% CI] | MRR [95% CI] | Δ recall@{k} vs baseline | article_recall@{k} | "
        "latency p50, ms | latency p95, ms | " + " | ".join(f"{s} p50" for s in steps) + " |"
    )
    lines = [
        f"Verified questions with a retrieval gold: n = {n_info['n']} "
        f"(real {n_info['real']}, synthetic {n_info['synthetic']}). "
        "Latency is measured on CPU after warm-up. Intervals: bootstrap over questions "
        f"(10,000 resamples); Δ: paired bootstrap against `{BASELINE_CELL}`, "
        "`*` — the interval excludes zero.",
        "",
        header,
        "|" + "---|" * (11 + len(steps)),
    ]
    for r in rows:
        lat = r["latency_ms"]
        total = lat.get("retrieval", {})
        cells = [
            r["chunking"],
            r["embeddings"],
            r["method"],
            str(r.get("chunks") or "—"),
            fmt(r.get("context_tokens"), 0),
            with_ci(r[f"recall@{k}"], r.get("recall_ci")),
            with_ci(r["mrr"], r.get("mrr_ci")),
            fmt_delta(r.get("vs_baseline")),
            fmt(r[f"article_recall@{k}"]),
            fmt(total.get("p50"), 1),
            fmt(total.get("p95"), 1),
            *[fmt(lat.get(s, {}).get("p50"), 1) for s in steps],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_name(split: str) -> str:
    return "retrieval_table.md" if split == "dev" else f"retrieval_table_{split}.md"


def stored_name(stamp: str, split: str) -> str:
    return f"{stamp}.json" if split == "dev" else f"{stamp}_{split}.json"


def latest_stored(out_dir: Path, split: str) -> Path:
    """Последний сохранённый прогон этой части набора (прежние файлы — это dev)."""
    runs = [
        p
        for p in sorted((out_dir / "retrieval").glob("2*.json"))
        if json.loads(p.read_text("utf-8")).get("split", "dev") == split
    ]
    if not runs:
        raise SystemExit(f"Нет сохранённых прогонов поиска для части {split}")
    return runs[-1]


def merge_runs(old_rows, old_runs, new_rows, new_runs):
    """Влить новые ячейки в сохранённый прогон: совпавшие заменяются, новые добавляются."""

    def key(row):
        return (row["chunking"], row["embeddings"], row["method"])

    fresh = {key(r): r for r in new_rows}
    rows = [fresh.pop(key(r), r) for r in old_rows] + list(fresh.values())
    return rows, {**old_runs, **new_runs}


def rerender(source: Path, out_dir: Path, k: int, seed: int) -> int:
    """Таблица с интервалами из сохранённого прогона: строки и выдача по вопросам уже есть."""
    data = json.loads(source.read_text("utf-8"))
    split = data.get("split", "dev")
    rows, runs = data["rows"], data["runs"]
    add_intervals(rows, runs, seed=seed)
    source.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
    table = render(rows, k, next(iter(runs.values()))["dataset"]["evaluated"])
    (out_dir / table_name(split)).write_text(
        f"<!-- generated by evals/retrieval_eval.py from {source.name} -->\n{table}", "utf-8"
    )
    print(table)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--only",
        default=None,
        help="оставить ячейки, где «нарезка/модель/метод» содержит любую из строк (через запятую)",
    )
    parser.add_argument("--results-dir", default="evals/results")
    parser.add_argument(
        "--split",
        default=None,
        choices=["dev", "test", "all"],
        help="часть набора (по умолчанию — eval.split из конфига: dev)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="влить выбранные ячейки в последний сохранённый прогон, не пересчитывая остальные",
    )
    parser.add_argument(
        "--from-json",
        metavar="PATH",
        help="пересобрать таблицу из сохранённого прогона, без поиска",
    )
    args = parser.parse_args()
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = load_config()
    if args.split:
        base.data["eval"]["split"] = args.split
    split = base.get("eval.split")
    k = int(base.get("eval.k"))
    seed = int(base.get("eval.seed"))
    if args.from_json:
        return rerender(Path(args.from_json), Path(args.results_dir), k, seed)
    dataset = load_dataset(base.path_of("eval.dataset"))

    tokenizer = build_tokenizer(base.get("chunking.tokenizer"))
    rows, runs = [], {}
    chunk_texts: dict[str, list[str]] = {}
    for chunking_name, embeddings, method in CELLS:
        version = f"{chunking_name}/{embeddings}/{method}"
        if args.only and not any(part in version for part in args.only.split(",")):
            continue
        config = cell_config(base, chunking_name, embeddings, method)

        chunks = build_store(config).all_chunks()
        texts = [c.text for c in chunks]
        if chunk_texts.setdefault(chunking_name, texts) != texts:
            raise RuntimeError(
                f"{version}: тексты чанков отличаются от других моделей той же нарезки — "
                "сравнение моделей было бы сравнением нарезок"
            )
        token_lengths = {c.chunk_id: len(tokenizer.offsets(c.text)) for c in chunks}

        log.info("Ячейка %s", version)
        result = EvalRunner(config, build_retriever(config)).run(dataset)
        runs[version] = result
        rows.append(
            {
                "chunking": chunking_name,
                "embeddings": embeddings,
                "method": method,
                **summarize(result, k, token_lengths),
            }
        )

    if not rows:
        print("Ни одной ячейки не выбрано.")
        return 1

    if args.merge:
        previous = latest_stored(Path(args.results_dir), split)
        old = json.loads(previous.read_text("utf-8"))
        rows, runs = merge_runs(old["rows"], old["runs"], rows, runs)
        log.info("Влито в %s", previous.name)
    add_intervals(rows, runs, seed=seed)
    n_info = next(iter(runs.values()))["dataset"]["evaluated"]
    table = render(rows, k, n_info)
    print("\n" + table)

    out_dir = Path(args.results_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / "retrieval").mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "retrieval" / stored_name(stamp, split)
    json_path.write_text(
        json.dumps({"split": split, "rows": rows, "runs": runs}, ensure_ascii=False, indent=2)
        + "\n",
        "utf-8",
    )
    (out_dir / table_name(split)).write_text(
        f"<!-- generated by evals/retrieval_eval.py from {json_path.name} -->\n{table}", "utf-8"
    )
    print(f"Записано: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
