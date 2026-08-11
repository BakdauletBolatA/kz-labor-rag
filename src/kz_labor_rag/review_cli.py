"""Отметка вопросов как отревьюированных.

    kzrag-review status                    # сколько проверено, сколько осталось
    kzrag-review mark syn_001 syn_002      # отметить проверенные
    kzrag-review mark --all                # отметить все готовые вопросы
    kzrag-review unmark syn_003            # снять отметку

Флаг ``reviewed_by_human`` — это гейт: пока хоть у одного вопроса он ``false``,
прогон eval останавливается. Ставить его руками в JSONL можно, но легко
опечататься и незаметно пометить проверенным то, что не смотрели, поэтому
правка идёт через схему датасета с перезаписью файла целиком.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from kz_labor_rag.config import ConfigError, load_config
from kz_labor_rag.eval.dataset import (
    DatasetError,
    EvalDataset,
    load_dataset,
    save_dataset,
)

log = logging.getLogger(__name__)


def _dataset_path(args):
    config = load_config(args.config)
    return args.dataset or config.path_of("eval.dataset")


def cmd_status(args) -> int:
    path = _dataset_path(args)
    dataset = load_dataset(path)
    ready = dataset.ready
    reviewed = [q for q in ready if q.reviewed_by_human]
    pending = [q for q in ready if not q.reviewed_by_human]

    print(f"Датасет: {path}")
    print(f"  готовых вопросов   : {len(ready)}")
    print(f"  отревьюировано     : {len(reviewed)}")
    print(f"  ждут ревью         : {len(pending)}")
    print(f"  черновых слотов    : {dataset.stats['draft_slots']}")

    if pending:
        shown = ", ".join(q.id for q in pending[:20])
        tail = f" и ещё {len(pending) - 20}" if len(pending) > 20 else ""
        print(f"\nНе отревьюированы: {shown}{tail}")
        print("\nОтметить проверенные: kzrag-review mark <id> [<id> ...]")
        print("Цитаты для ревью:     evals/datasets/REVIEW.md")
        return 1

    print("\nВсе готовые вопросы отревьюированы.")
    return 0


def _apply(args, *, value: bool) -> int:
    path = _dataset_path(args)
    dataset = load_dataset(path)
    by_id = {q.id: q for q in dataset}

    if args.all:
        targets = [q.id for q in dataset.ready]
    else:
        targets = list(args.ids)
        if not targets:
            print("Нужны id вопросов или --all", file=sys.stderr)
            return 1

    unknown = [qid for qid in targets if qid not in by_id]
    if unknown:
        print(f"Нет таких вопросов: {', '.join(unknown)}", file=sys.stderr)
        return 1

    drafts = [qid for qid in targets if by_id[qid].is_draft]
    if drafts:
        # Черновой слот нечего ревьюировать: вопроса ещё нет.
        print(
            f"Это черновые слоты, ревьюировать нечего: {', '.join(drafts)}",
            file=sys.stderr,
        )
        return 1

    changed = [qid for qid in targets if by_id[qid].reviewed_by_human != value]
    updated = tuple(
        replace(q, reviewed_by_human=value) if q.id in set(targets) and not q.is_draft else q
        for q in dataset
    )
    save_dataset(EvalDataset(questions=updated, path=dataset.path), path)

    # «0 из 3» читается как отказ, хотя означает «уже было сделано».
    # Разделяем новые отметки и повтор, иначе повторный запуск выглядит сбоем.
    already = len(targets) - len(changed)
    if value:
        line = f"новых отметок: {len(changed)}"
        if already:
            line += f", уже было отмечено: {already}"
    else:
        line = f"снято отметок: {len(changed)}"
        if already:
            line += f", и так не были отмечены: {already}"
    print(line)

    remaining = [q.id for q in EvalDataset(questions=updated).ready if not q.reviewed_by_human]
    if remaining:
        print(f"Осталось без ревью: {len(remaining)}")
    else:
        print("Все готовые вопросы отревьюированы — гейт по ревью пройден.")
    return 0


def cmd_mark(args) -> int:
    return _apply(args, value=True)


def cmd_unmark(args) -> int:
    return _apply(args, value=False)


def main(argv: list[str] | None = None) -> int:
    # Общие флаги повторяются в каждой подкоманде: argparse не принимает
    # опции верхнего уровня после подкоманды, а «kzrag-review mark syn_001
    # --dataset …» — ровно то, как их напишут.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None)
    common.add_argument("--dataset", default=None)

    parser = argparse.ArgumentParser(
        prog="kzrag-review", description=__doc__, parents=[common]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="сколько проверено", parents=[common]).set_defaults(
        func=cmd_status
    )

    p_mark = sub.add_parser("mark", help="отметить вопросы проверенными", parents=[common])
    p_mark.add_argument("ids", nargs="*")
    p_mark.add_argument("--all", action="store_true", help="все готовые вопросы")
    p_mark.set_defaults(func=cmd_mark)

    p_unmark = sub.add_parser("unmark", help="снять отметку", parents=[common])
    p_unmark.add_argument("ids", nargs="*")
    p_unmark.add_argument("--all", action="store_true")
    p_unmark.set_defaults(func=cmd_unmark)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ConfigError, DatasetError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
