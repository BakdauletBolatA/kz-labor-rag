"""Управление индексом.

kzrag-index build              # построить (переиспользуя кэш эмбеддингов)
kzrag-index build --rebuild    # снести схему и построить заново
kzrag-index status             # чем построен текущий индекс
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from kz_labor_rag.config import ConfigError, load_config
from kz_labor_rag.corpus.parser import ParseError
from kz_labor_rag.indexer import build_index, index_mismatch
from kz_labor_rag.retrieval.factory import build_store
from kz_labor_rag.retrieval.store import StoreError

log = logging.getLogger(__name__)


def cmd_build(args) -> int:
    config = load_config(args.config)
    report = build_index(config, rebuild=args.rebuild)
    print("Индекс построен:")
    for line in report.as_lines():
        print(f"  {line}")
    return 0


def cmd_status(args) -> int:
    config = load_config(args.config)
    store = build_store(config)
    count = store.count()
    meta = store.read_meta()

    print(f"Чанков в таблице: {count}")
    if meta is None:
        print("Индекс не построен. Постройте: kzrag-index build")
        return 1

    for key in (
        "version",
        "corpus_edition_date",
        # Вердикт о расхождении выносится по хешу файла и версии парсера,
        # поэтому их видно рядом: иначе сообщение «другая редакция корпуса»
        # не с чем сопоставить глазами.
        "corpus_sha256",
        "parser_version",
        "embeddings_model",
        "chunking_signature",
        "config_fingerprint",
        "built_at",
    ):
        print(f"  {key:22} {meta.get(key)}")
    print(f"  {'chunking':22} {json.dumps(meta.get('chunking'), ensure_ascii=False)}")

    if problem := index_mismatch(config, store):
        print(f"\nРасхождение с конфигом: {problem}", file=sys.stderr)
        return 1
    print("\nИндекс соответствует текущему конфигу.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kzrag-index", description=__doc__)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="построить индекс")
    p_build.add_argument("--rebuild", action="store_true", help="снести таблицу и построить с нуля")
    p_build.set_defaults(func=cmd_build)

    sub.add_parser("status", help="чем построен индекс").set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ConfigError, ParseError, StoreError, FileNotFoundError, NotImplementedError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
