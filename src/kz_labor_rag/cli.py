"""CLI eval-харнесса.

kzrag-eval validate            # проверить датасет: схема и сколько вопросов проверено
kzrag-eval run                 # прогнать eval и записать JSON
kzrag-eval show <result.json>  # показать агрегаты и проваленные вопросы
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kz_labor_rag.config import Config, ConfigError, load_config
from kz_labor_rag.eval.compare import IncomparableRunsError, build_table
from kz_labor_rag.eval.dataset import DatasetError, load_dataset
from kz_labor_rag.eval.factory import build_generator, build_judge
from kz_labor_rag.eval.prompts import PromptRegistryError
from kz_labor_rag.eval.runner import EvalRunner, NoVerifiedQuestionsError, save_result
from kz_labor_rag.indexer import index_mismatch
from kz_labor_rag.retrieval.factory import build_retriever as _build_retriever
from kz_labor_rag.retrieval.factory import build_store
from kz_labor_rag.retrieval.store import StoreError
from kz_labor_rag.types import Retriever


def build_retriever(config: Config) -> Retriever:
    """Собрать поиск по конфигу и убедиться, что индекс ему соответствует."""
    if problem := index_mismatch(config, build_store(config)):
        raise StoreError(
            f"{problem}\n"
            "Прогон на индексе, не соответствующем конфигу, даёт правдоподобные, "
            "но бессмысленные числа."
        )
    return _build_retriever(config)


def cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = load_dataset(args.dataset or config.path_of("eval.dataset"))

    print(f"Датасет: {dataset.path}")
    for key, value in dataset.stats.items():
        print(f"  {key:24} {value}")

    verified = dataset.verified
    print(f"\nПроверено человеком: {len(verified)} из {len(dataset.ready)} готовых.")
    if not verified:
        print("Метрики считать не по чему: нет ни одного проверенного вопроса.")
        return 1
    print("Метрики будут посчитаны только по проверенным вопросам.")
    return 0


def assert_search_depth_matches(config) -> None:
    """eval.k и retrieval.top_k — одна и та же глубина поиска, названная дважды.

    Прогон ищет с ``eval.k``, а ``/search``, ``/ask`` и отладочный CLI — с
    ``retrieval.top_k``. Разъедься они, и метрики описывали бы глубину, на
    которой не работает ни одна точка входа: recall@10 при выдаче в пять
    чанков пользователю. Ошибка тихая — обе цифры выглядят осмысленно
    по отдельности.
    """
    eval_k = int(config.get("eval.k"))
    serve_k = int(config.get("retrieval.top_k"))
    if eval_k != serve_k:
        raise ConfigError(
            f"eval.k = {eval_k}, а retrieval.top_k = {serve_k}. Это одна и та же "
            "глубина поиска: прогон измерял бы выдачу, которой не отдают ни API, "
            "ни kzrag-search. Приведите значения к одному."
        )


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    assert_search_depth_matches(config)
    dataset = load_dataset(args.dataset or config.path_of("eval.dataset"))

    # Отсутствие ключа API отключает генерацию и судью с предупреждением
    # в лог, но не мешает посчитать все метрики поиска.
    generator = build_generator(config)
    judge = build_judge(config, generator=generator)

    runner = EvalRunner(
        config=config,
        retriever=build_retriever(config),
        generator=generator,
        judge=judge,
    )

    try:
        result = runner.run(dataset)
    except NoVerifiedQuestionsError as exc:
        print(exc, file=sys.stderr)
        return 1

    path = save_result(result, args.results_dir or config.path_of("eval.results_dir"))
    print(f"Результат записан: {path}")
    _print_aggregates(result)
    return 0


def _print_aggregates(result: dict) -> None:
    print(f"\nВерсия: {result['version']}  ({result.get('description', '')})")
    if evaluated := result.get("dataset", {}).get("evaluated"):
        print(
            f"Посчитано по {evaluated['n']} проверенным вопросам "
            f"(real {evaluated['real']}, synthetic {evaluated['synthetic']}); "
            f"без ответа в кодексе: {evaluated.get('unanswerable', 0)}; "
            f"непроверенных пропущено: {evaluated['skipped_unverified']}"
        )
    for lang, agg in result["aggregates"]["by_language"].items():
        if not agg.get("n"):
            continue
        print(f"\n[{lang}]  n={agg['n']}")
        for key, value in agg.items():
            if key in ("n", "retrieval_failures") or value is None:
                continue
            print(f"  {key:24} {value:.3f}" if isinstance(value, float) else f"  {key:24} {value}")
        if failures := agg.get("retrieval_failures"):
            print(f"  провалы поиска ({len(failures)}): {', '.join(failures[:15])}")


def cmd_compare(args: argparse.Namespace) -> int:
    """Построить таблицу «до/после» — или отказаться, если прогоны несравнимы."""
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    try:
        print(build_table(before, after, language=args.language))
    except IncomparableRunsError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    _print_aggregates(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kzrag-eval", description=__doc__)
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="проверить датасет и число проверенных вопросов")
    p_validate.add_argument("--dataset", default=None)
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="прогнать eval и записать результат")
    p_run.add_argument("--dataset", default=None)
    p_run.add_argument("--results-dir", default=None)
    p_run.set_defaults(func=cmd_run)

    p_compare = sub.add_parser("compare", help="таблица «до/после» по двум прогонам")
    p_compare.add_argument("before")
    p_compare.add_argument("after")
    p_compare.add_argument("--language", default=None, help="срез: ru или kk")
    p_compare.set_defaults(func=cmd_compare)

    p_show = sub.add_parser("show", help="показать агрегаты из записанного JSON")
    p_show.add_argument("result")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (
        ConfigError,
        DatasetError,
        PromptRegistryError,
        StoreError,
        NotImplementedError,
    ) as exc:
        # Ожидаемые состояния проекта — незаполненный конфиг, отсутствующий
        # датасет, ещё не реализованный поиск. Трейсбек здесь только мешает
        # прочитать, что именно нужно сделать.
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
