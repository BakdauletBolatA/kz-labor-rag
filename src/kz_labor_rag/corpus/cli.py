"""CLI парсера корпуса.

kzrag-corpus stats            # разобрать и показать статистику
kzrag-corpus show 54          # показать разобранную статью
kzrag-corpus quote 62 "в течение пяти"   # вырезать цитату по якорю для evidence
kzrag-corpus dump             # выгрузить разбор в JSON
kzrag-corpus check-dataset    # сверить эталон датасета с текстом кодекса
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kz_labor_rag.config import ConfigError, load_config
from kz_labor_rag.corpus.evidence import extract_evidence
from kz_labor_rag.corpus.parser import ParseError, parse_file
from kz_labor_rag.eval.dataset import DatasetError, load_dataset, validate_against_corpus
from kz_labor_rag.types import article_sort_key, normalize_article

log = logging.getLogger(__name__)


def _load(args) -> tuple:
    config = load_config(args.config)
    raw = args.raw or config.path_of("corpus.raw_html")
    if not Path(raw).exists():
        raise ParseError(
            f"сырой HTML не найден: {raw}\n"
            "Он не коммитится в репозиторий. Команда для скачивания — "
            "в data/raw/SOURCE.md."
        )
    code = parse_file(raw, strip_amendment_notes=config.get("corpus.strip_amendment_notes"))
    return config, code


def cmd_stats(args) -> int:
    _, code = _load(args)
    print(f"Редакция (по метке ИПС): {code.edition_date}")
    print(f"Версия парсера:          {code.parser_version}")
    for key, value in code.stats.items():
        print(f"  {key:22} {value}")

    repealed = [a.number for a in code if a.is_repealed]
    future = [a.number for a in code if a.has_future_edition]
    print(f"\nИсключённые статьи ({len(repealed)}): {', '.join(repealed)}")
    print(f"Статьи с объявленной будущей редакцией ({len(future)}): {', '.join(future)}")
    print(
        "\nСтатьи выше требуют внимания при разметке эталона: исключённые нельзя "
        "ставить в required_articles, а у статей с будущей редакцией «правильная» "
        "формулировка зависит от даты."
    )
    return 0


def cmd_show(args) -> int:
    _, code = _load(args)
    number = normalize_article(args.article)
    article = code.by_number.get(number)
    if article is None:
        near = sorted(code.by_number, key=article_sort_key)[:5]
        print(f"Статья {number} не найдена. Например, есть: {', '.join(near)}", file=sys.stderr)
        return 1

    print(article.heading)
    print(f"  {article.part} / {article.section} / {article.chapter}")
    if article.is_repealed:
        print("  [ИСКЛЮЧЕНА]")
    if article.has_future_edition:
        print("  [объявлена будущая редакция]")
    print()
    for clause in article.clauses:
        label = f"п. {clause.number}." if clause.number else "(без номера)"
        print(f"{label} {clause.text}\n")
    for note in article.amendment_notes:
        print(f"  сноска: {note}")
    for note in article.izpi_notes:
        print(f"  ИЗПИ:   {note}")
    return 0


def cmd_quote(args) -> int:
    """Вырезать цитату по якорю — для разметки реальных и казахских вопросов.

    Синтетические вопросы задаются якорями в SPECS, и цитату им вырезает
    сборка. Реальные и казахские пишутся в датасет руками, и цитату там иначе
    пришлось бы перенабирать из кодекса — то есть ровно то, чего вся схема
    избегает: перенабранная цитата отличается от текста статьи одним символом,
    и обоснование разметки становится неверным.

    Печатается тот же фрагмент, что положила бы в датасет сборка: правило
    вырезания одно на всех, в ``corpus/evidence.py``.
    """
    _, code = _load(args)
    number = normalize_article(args.article)
    article = code.by_number.get(number)
    if article is None:
        near = sorted(code.by_number, key=article_sort_key)[:5]
        print(f"Статья {number} не найдена. Например, есть: {', '.join(near)}", file=sys.stderr)
        return 1
    if article.is_repealed:
        # Статья без нормативного текста непроходима в принципе, и в метриках
        # это выглядит как провал поиска, а не как ошибка разметки.
        print(f"Статья {number} исключена из кодекса, цитировать нечего.", file=sys.stderr)
        return 1

    try:
        quote = extract_evidence(article.full_text, args.anchor)
    except ValueError:
        print(
            f"Якорь не найден в тексте ст. {number}: {args.anchor!r}\n"
            f"Посмотреть текст целиком: kzrag-corpus show {number}",
            file=sys.stderr,
        )
        return 1

    print(article.heading)
    if article.has_future_edition:
        print("  [объявлена будущая редакция: формулировка зависит от даты]")
    if clauses := article.clause_numbers:
        print(f"  пункты: {', '.join(clauses)}")
    print(f"\n> {quote}\n")
    print("в evidence:")
    print(json.dumps({"article": number, "quote": quote}, ensure_ascii=False))
    return 0


def cmd_dump(args) -> int:
    config, code = _load(args)
    out = Path(args.out or config.root / "data/processed/labor_code.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "parser_version": code.parser_version,
        "edition_date": code.edition_date,
        "meta": code.meta,
        "stats": code.stats,
        "articles": [
            {
                "number": a.number,
                "title": a.title,
                "part": a.part,
                "section": a.section,
                "chapter": a.chapter,
                "is_repealed": a.is_repealed,
                "has_future_edition": a.has_future_edition,
                "clauses": [{"number": c.number, "text": c.text} for c in a.clauses],
                "amendment_notes": list(a.amendment_notes),
                "izpi_notes": list(a.izpi_notes),
            }
            for a in code
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записано: {out} ({len(code)} статей)")
    return 0


def cmd_check_dataset(args) -> int:
    """Сверить разметку эталона с текстом кодекса.

    Ловит три класса ошибок разметки: ссылку на несуществующую статью,
    выдуманную цитату и пункт, которого в статье нет. Плюс отдельно —
    ссылку на исключённую статью: она существует, но текста не имеет,
    и вопрос с ней непроходим в принципе.
    """
    config, code = _load(args)
    dataset = load_dataset(args.dataset or config.path_of("eval.dataset"))
    report = validate_against_corpus(
        dataset,
        code.article_texts(),
        {a.number: a.clause_numbers for a in code},
    )

    repealed = {a.number for a in code if a.is_repealed}
    hits_repealed = [(q.id, num) for q in dataset for num in q.required_articles if num in repealed]

    problems = 0
    if report.missing_articles:
        problems += len(report.missing_articles)
        print("Ссылки на несуществующие статьи:")
        for qid, num in report.missing_articles:
            print(f"  {qid}: статья {num}")
    if hits_repealed:
        problems += len(hits_repealed)
        print("\nСсылки на исключённые статьи (текста нет, вопрос непроходим):")
        for qid, num in hits_repealed:
            print(f"  {qid}: статья {num}")
    if report.quote_not_found:
        problems += len(report.quote_not_found)
        print("\nЦитата не найдена в тексте обязательных статей:")
        for qid in report.quote_not_found:
            print(f"  {qid}")
    if report.missing_clauses:
        problems += len(report.missing_clauses)
        print("\npreferred_clause указывает на несуществующий пункт:")
        for qid, ref in report.missing_clauses:
            print(f"  {qid}: {ref}")

    if problems:
        print(f"\nВсего проблем разметки: {problems}")
        return 1

    # Печатать длину файла нельзя: у черновых слотов разметки нет, валидатор
    # их пропускает, и отчёт про «60 вопросов» обещал бы больше проверенного,
    # чем проверено на самом деле.
    skipped = len(dataset) - len(dataset.ready)
    tail = f" ({skipped} черновых слотов пропущено)" if skipped else ""
    print(f"Разметка сверена с корпусом: {len(dataset.ready)} вопросов, расхождений нет{tail}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kzrag-corpus", description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw", default=None, help="путь к сохранённому HTML")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="статистика разбора").set_defaults(func=cmd_stats)

    p_show = sub.add_parser("show", help="показать статью")
    p_show.add_argument("article")
    p_show.set_defaults(func=cmd_show)

    p_quote = sub.add_parser("quote", help="вырезать цитату по якорю для evidence")
    p_quote.add_argument("article")
    p_quote.add_argument("anchor", help="фрагмент текста статьи, вокруг которого резать")
    p_quote.set_defaults(func=cmd_quote)

    p_dump = sub.add_parser("dump", help="выгрузить разбор в JSON")
    p_dump.add_argument("--out", default=None)
    p_dump.set_defaults(func=cmd_dump)

    p_check = sub.add_parser("check-dataset", help="сверить эталон датасета с корпусом")
    p_check.add_argument("--dataset", default=None)
    p_check.set_defaults(func=cmd_check_dataset)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ConfigError, DatasetError, ParseError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
