"""Ревью тестового набора.

    python evals/review.py                 # пройти по непроверенным вопросам
    python evals/review.py syn_004 new_001 # только эти вопросы
    python evals/review.py --status        # сколько проверено, по типам

Для каждого вопроса показан полный текст пунктов, на которые размечен ответ.
Команды: [v] верно, [e] исправить пункты, [d] удалить, [s] пропустить,
[q] выйти. Файл сохраняется после каждого действия.
"""

from __future__ import annotations

import argparse

from kz_labor_rag.config import load_config, load_env_file
from kz_labor_rag.corpus.parser import parse_file
from kz_labor_rag.eval.review import ReviewSession


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("ids", nargs="*", help="id вопросов; по умолчанию — все непроверенные")
    parser.add_argument("--status", action="store_true", help="только показать прогресс")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    load_env_file()

    config = load_config()
    code = parse_file(
        config.path_of("corpus.raw_html"),
        strip_amendment_notes=bool(config.get("corpus.strip_amendment_notes")),
    )
    session = ReviewSession(args.dataset or config.path_of("eval.dataset"), code)
    if args.status:
        print(session.status())
        return 0
    try:
        session.run(args.ids)
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано. Всё, что отмечено до этого, сохранено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
