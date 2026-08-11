"""Отладка выдачи поиска.

    kzrag-search "меня уволили в отпуске"     # свободный запрос
    kzrag-search --question syn_001           # вопрос из датасета с разбором промаха
    kzrag-search --failures                   # все вопросы, где эталон не найден
    kzrag-search --article 54                 # как статья разрезана на чанки

Инструмент существует ради пункта 5 плана: список гипотез улучшения
составляется после того, как видно, на каких вопросах baseline проваливается
и почему. Поэтому здесь показываются сырые скоры и id чанков, а для вопросов
датасета — на каком ранге оказалась каждая обязательная статья и что оказалось
выше неё.
"""

from __future__ import annotations

import argparse
import logging
import sys

from kz_labor_rag.config import ConfigError, load_config
from kz_labor_rag.eval.dataset import DatasetError, EvalQuestion, load_dataset
from kz_labor_rag.eval.metrics import rank_articles
from kz_labor_rag.indexer import index_mismatch
from kz_labor_rag.retrieval.factory import build_retriever, build_store
from kz_labor_rag.retrieval.store import StoreError

log = logging.getLogger(__name__)

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def _plain(text: str) -> str:
    return text if sys.stdout.isatty() else ""


def show_hits(hits, *, required: set[str] | None = None, preview: int) -> None:
    required = required or set()
    for hit in hits:
        mark = ""
        if required:
            mark = (
                f" {_plain(GREEN)}◄ обязательная{_plain(RESET)}"
                if set(hit.articles) & required
                else ""
            )
        articles = ", ".join(f"ст. {a}" for a in hit.articles)
        clauses = ", ".join(f"{a}/{c}" for a, c in hit.chunk.spans)
        print(f"  #{hit.rank}  score={hit.score:+.4f}  {hit.chunk_id}  [{articles}]{mark}")
        print(f"      пункты: {clauses or '—'}")
        print(f"      {_plain(DIM)}{hit.text[:preview].replace(chr(10), ' ')}…{_plain(RESET)}")


def diagnose(question: EvalQuestion, hits, k: int) -> None:
    """Разобрать, что случилось с обязательными статьями вопроса."""
    ranked = rank_articles(hits)
    required = list(question.required_articles)

    print(f"\n  Эталон: {', '.join('ст. ' + a for a in required)}")
    if question.acceptable_articles:
        print(f"  Допустимые: {', '.join('ст. ' + a for a in question.acceptable_articles)}")

    for article in required:
        if article in ranked:
            position = ranked.index(article) + 1
            status = f"в топ-{k}" if position <= k else f"ниже топ-{k}"
            print(f"  ст. {article}: ранг {position} ({status})")
        else:
            print(f"  {_plain(RED)}ст. {article}: НЕ НАЙДЕНА вовсе{_plain(RESET)}")

    missing = [a for a in required if a not in ranked[:k]]
    if missing:
        noise = [
            a for a in ranked[:k] if a not in required and a not in question.acceptable_articles
        ]
        print(f"\n  Вытеснили эталон: {', '.join('ст. ' + a for a in noise) or '—'}")

    if question.preferred_clause:
        covered = any(h.covers(question.preferred_clause) for h in hits[:k])
        flag = "покрыт" if covered else "НЕ покрыт"
        print(f"  Точный пункт {question.preferred_clause}: {flag}")


def _retriever(args):
    config = load_config(args.config)
    store = build_store(config)
    if problem := index_mismatch(config, store):
        print(f"Внимание: {problem}", file=sys.stderr)
    return config, build_retriever(config)


def cmd_search(args) -> int:
    config, retriever = _retriever(args)
    k = args.k or int(config.get("retrieval.top_k"))

    if args.article:
        store = build_store(config)
        chunks = store.by_article(args.article)
        print(f"Статья {args.article}: попала в {len(chunks)} чанков")
        for chunk in chunks:
            others = [a for a in chunk.articles if a != args.article]
            tail = f"  (вместе со ст. {', '.join(others)})" if others else ""
            print(f"  {chunk.chunk_id}  пункты: {', '.join(chunk.clauses) or '—'}{tail}")
            print(f"      {_plain(DIM)}{chunk.text[:160]}…{_plain(RESET)}")
        return 0

    if args.failures or args.question:
        dataset = load_dataset(args.dataset or config.get("eval.dataset"))
        selected = [q for q in dataset.ready if not args.question or q.id == args.question]
        if args.question and not selected:
            print(f"Вопроса {args.question} нет в датасете", file=sys.stderr)
            return 1

        failures = 0
        for question in selected:
            hits = list(retriever.search(question.question, k))
            found = set(rank_articles(hits)[:k]) & set(question.required_articles)
            if args.failures and found:
                continue
            failures += 1
            print(f"\n{'=' * 76}\n{question.id}  [{', '.join(question.tags)}]")
            print(f"  {question.question}")
            show_hits(hits, required=set(question.required_articles), preview=args.preview)
            diagnose(question, hits, k)

        if args.failures:
            print(f"\n{'=' * 76}")
            print(
                f"Вопросов без единой обязательной статьи в топ-{k}: {failures} из {len(selected)}"
            )
        return 0

    if not args.query:
        print("Нужен запрос, --question, --failures или --article", file=sys.stderr)
        return 1

    hits = list(retriever.search(args.query, k))
    print(f"Запрос: {args.query}\nВерсия пайплайна: {retriever.version}\n")
    show_hits(hits, preview=args.preview)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kzrag-search", description=__doc__)
    parser.add_argument("query", nargs="?", help="свободный поисковый запрос")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("-k", type=int, default=None, help="сколько чанков показать")
    parser.add_argument("--question", help="id вопроса из датасета, например syn_001")
    parser.add_argument(
        "--failures",
        action="store_true",
        help="показать только вопросы, где ни одна обязательная статья не попала в топ-k",
    )
    parser.add_argument("--article", help="показать, как статья разрезана на чанки")
    parser.add_argument("--preview", type=int, default=200, help="длина превью текста чанка")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return cmd_search(args)
    except (ConfigError, DatasetError, StoreError, NotImplementedError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
