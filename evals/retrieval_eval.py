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
import copy
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kz_labor_rag.cli import build_retriever
from kz_labor_rag.config import Config, load_config
from kz_labor_rag.eval.dataset import load_dataset
from kz_labor_rag.eval.runner import EvalRunner
from kz_labor_rag.indexer import build_index, index_mismatch
from kz_labor_rag.retrieval.factory import build_store

log = logging.getLogger("retrieval_eval")

CHUNKINGS: dict[str, dict] = {
    "fixed512": {"chunking.strategy": "fixed_tokens"},
}

EMBEDDINGS: dict[str, dict] = {
    "e5-base": {},
    "bge-m3": {
        "embeddings.model": "BAAI/bge-m3",
        "embeddings.dimensions": 1024,
        "embeddings.query_prefix": "",
        "embeddings.passage_prefix": "",
        "embeddings.max_sequence_length": 8192,
    },
}

METHODS: dict[str, dict] = {
    "dense": {"retrieval.implementation": "dense"},
    "bm25-lemma": {"retrieval.implementation": "bm25", "retrieval.bm25.analyzer": "lemma"},
    "bm25-stem": {"retrieval.implementation": "bm25", "retrieval.bm25.analyzer": "stem"},
    "hybrid": {"retrieval.implementation": "hybrid"},
    "dense+rerank": {"retrieval.implementation": "dense", "retrieval.reranker.enabled": True},
    "hybrid+rerank": {"retrieval.implementation": "hybrid", "retrieval.reranker.enabled": True},
}

# Модель эмбеддингов влияет только на dense-часть; сравнивать её имеет смысл
# на чистом dense, остальные методы идут на базовой модели.
CELLS: list[tuple[str, str]] = [("e5-base", m) for m in METHODS] + [("bge-m3", "dense")]


def derive(base: Config, *overrides: dict, version: str, table: str) -> Config:
    data = copy.deepcopy(base.data)
    for block in overrides:
        for dotted, value in block.items():
            node = data
            *path, last = dotted.split(".")
            for key in path:
                node = node[key]
            node[last] = value
    data["version"] = version
    data["vector_store"]["table"] = table
    # Генерация и судья здесь не участвуют: таблица только про поиск.
    data["generation"]["enabled"] = False
    data["judge"]["enabled"] = False
    return Config(data=data, path=base.path, root=base.root)


def ensure_index(config: Config, budget_config: Config) -> None:
    store = build_store(config)
    if store.count() and not index_mismatch(config, store):
        return
    log.info("Строится индекс %s", config.get("vector_store.table"))
    build_index(config, store=store, budget_config=budget_config)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(result: dict, k: int) -> dict:
    agg = result["aggregates"]["primary"]
    latencies: dict[str, list[float]] = {}
    for question in result["questions"]:
        for step, ms in question["latency_ms"].items():
            if step != "generation":
                latencies.setdefault(step, []).append(ms)
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
        "chunking_signature": result["index"].get("chunking_signature"),
        "failures": agg.get("retrieval_failures", []),
    }


def fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(rows: list[dict], k: int, n_info: dict) -> str:
    steps = ["dense", "bm25", "fusion", "rerank"]
    header = (
        f"| chunking | embeddings | method | recall@{k} | MRR | article_recall@{k} | "
        "latency p50, ms | latency p95, ms | " + " | ".join(f"{s} p50" for s in steps) + " |"
    )
    lines = [
        f"Verified questions with a retrieval gold: n = {n_info['n']} "
        f"(real {n_info['real']}, synthetic {n_info['synthetic']}). "
        "Latency is measured on CPU after warm-up.",
        "",
        header,
        "|" + "---|" * (8 + len(steps)),
    ]
    for r in rows:
        lat = r["latency_ms"]
        total = lat.get("retrieval", {})
        cells = [
            r["chunking"],
            r["embeddings"],
            r["method"],
            fmt(r[f"recall@{k}"]),
            fmt(r["mrr"]),
            fmt(r[f"article_recall@{k}"]),
            fmt(total.get("p50"), 1),
            fmt(total.get("p95"), 1),
            *[fmt(lat.get(s, {}).get("p50"), 1) for s in steps],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--only", default=None, help="оставить ячейки, где метод содержит строку")
    parser.add_argument("--results-dir", default="evals/results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = load_config()
    k = int(base.get("eval.k"))
    dataset = load_dataset(base.path_of("eval.dataset"))

    rows, runs = [], {}
    chunk_texts: dict[str, list[str]] = {}
    for chunking, chunking_overrides in CHUNKINGS.items():
        budget_config = derive(base, chunking_overrides, version="budget", table="unused")
        for embeddings, method in CELLS:
            if args.only and args.only not in method:
                continue
            table = f"exp_{chunking}_{embeddings}".replace("-", "_").replace(".", "_")
            version = f"{chunking}/{embeddings}/{method}"
            config = derive(
                base,
                chunking_overrides,
                EMBEDDINGS[embeddings],
                METHODS[method],
                version=version,
                table=table,
            )
            ensure_index(config, budget_config)

            texts = [c.text for c in build_store(config).all_chunks()]
            if chunk_texts.setdefault(chunking, texts) != texts:
                raise RuntimeError(
                    f"{version}: тексты чанков отличаются от других моделей той же нарезки — "
                    "сравнение моделей было бы сравнением нарезок"
                )

            log.info("Ячейка %s", version)
            result = EvalRunner(config, build_retriever(config)).run(dataset)
            runs[version] = result
            rows.append(
                {
                    "chunking": chunking,
                    "embeddings": embeddings,
                    "method": method,
                    **summarize(result, k),
                }
            )

    if not rows:
        print("Ни одной ячейки не выбрано.")
        return 1

    n_info = next(iter(runs.values()))["dataset"]["evaluated"]
    table = render(rows, k, n_info)
    print("\n" + table)

    out_dir = Path(args.results_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / "retrieval").mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "retrieval" / f"{stamp}.json"
    json_path.write_text(
        json.dumps({"rows": rows, "runs": runs}, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    (out_dir / "retrieval_table.md").write_text(
        f"<!-- generated by evals/retrieval_eval.py from {json_path.name} -->\n{table}", "utf-8"
    )
    print(f"Записано: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
