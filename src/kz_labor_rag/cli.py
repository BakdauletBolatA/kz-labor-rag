"""CLI eval-харнесса.

    kzrag-eval validate            # проверить датасет: схема, гейт, сверка с корпусом
    kzrag-eval run                 # прогнать eval и записать JSON
    kzrag-eval show <result.json>  # показать агрегаты и проваленные вопросы
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kz_labor_rag.config import Config, ConfigError, load_config
from kz_labor_rag.eval.dataset import CompletenessRule, DatasetError, load_dataset
from kz_labor_rag.eval.generator import AnthropicGenerator, DisabledGenerator
from kz_labor_rag.eval.judge import AnthropicJudge, DisabledJudge
from kz_labor_rag.eval.prompts import PromptRegistryError
from kz_labor_rag.eval.runner import DatasetNotReadyError, EvalRunner, save_result
from kz_labor_rag.types import Retriever


def build_retriever(config: Config) -> Retriever:
    """Собрать поиск по конфигу.

    Пока реализаций нет: харнесс по плану собирается раньше RAG. Сообщение об
    ошибке должно объяснять это, а не выглядеть поломкой.
    """
    backend = config.get_or("retrieval.implementation", None)
    if backend is None:
        raise NotImplementedError(
            "Поиск ещё не реализован — это ожидаемо: харнесс собирается до RAG.\n"
            "Задайте retrieval.implementation в конфиге, когда появится baseline (пункт 4 плана).\n"
            "Сейчас доступна команда: kzrag-eval validate"
        )
    raise NotImplementedError(f"неизвестная реализация поиска: {backend!r}")


def build_generator(config: Config):
    if not config.get("generation.enabled"):
        return DisabledGenerator()
    provider = config.get("generation.provider")
    if provider != "anthropic":
        raise NotImplementedError(f"провайдер генерации '{provider}' не поддержан")
    return AnthropicGenerator(
        model=config.get("generation.model"),
        prompt_id=config.get("generation.prompt_id"),
        prompt_version=config.get("generation.prompt_version"),
        max_tokens=int(config.get("generation.max_tokens")),
        temperature=float(config.get("generation.temperature")),
    )


def build_judge(config: Config):
    if not config.get("judge.enabled"):
        return DisabledJudge()
    provider = config.get("judge.provider")
    if provider != "anthropic":
        raise NotImplementedError(f"провайдер судьи '{provider}' не поддержан")
    return AnthropicJudge(
        model=config.get("judge.model"),
        prompt_id=config.get("judge.prompt_id"),
        prompt_version=config.get("judge.prompt_version"),
        max_tokens=int(config.get("judge.max_tokens")),
        temperature=float(config.get("judge.temperature")),
    )


def cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = load_dataset(args.dataset or config.get("eval.dataset"))

    print(f"Датасет: {dataset.path}")
    for key, value in dataset.stats.items():
        print(f"  {key:24} {value}")

    rules = config.section("eval")["completeness"]
    problems = CompletenessRule(
        min_ru=int(rules["min_ru"]),
        min_kk=int(rules["min_kk"]),
        min_real=int(rules["min_real"]),
        require_human_review=bool(rules["require_human_review"]),
    ).violations(dataset)

    if problems:
        print("\nГейт готовности НЕ пройден:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nBaseline запускать нельзя, цифры в EVALUATION.md писать нельзя.")
        return 1

    print("\nГейт готовности пройден: датасет укомплектован.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = load_dataset(args.dataset or config.get("eval.dataset"))

    runner = EvalRunner(
        config=config,
        retriever=build_retriever(config),
        generator=build_generator(config),
        judge=build_judge(config),
    )

    try:
        result = runner.run(dataset, enforce_gate=not args.no_gate)
    except DatasetNotReadyError as exc:
        print(exc, file=sys.stderr)
        return 1

    path = save_result(result, args.results_dir or config.get("eval.results_dir"))
    print(f"Результат записан: {path}")
    _print_aggregates(result)
    return 0


def _print_aggregates(result: dict) -> None:
    print(f"\nВерсия: {result['version']}  ({result.get('description', '')})")
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


def cmd_show(args: argparse.Namespace) -> int:
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    _print_aggregates(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kzrag-eval", description=__doc__)
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="проверить датасет и гейт готовности")
    p_validate.add_argument("--dataset", default=None)
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="прогнать eval и записать результат")
    p_run.add_argument("--dataset", default=None)
    p_run.add_argument("--results-dir", default=None)
    p_run.add_argument(
        "--no-gate",
        action="store_true",
        help="прогнать на неукомплектованном датасете (результат нельзя писать в EVALUATION.md)",
    )
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show", help="показать агрегаты из записанного JSON")
    p_show.add_argument("result")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DatasetError, PromptRegistryError, NotImplementedError) as exc:
        # Ожидаемые состояния проекта — незаполненный конфиг, отсутствующий
        # датасет, ещё не реализованный поиск. Трейсбек здесь только мешает
        # прочитать, что именно нужно сделать.
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
