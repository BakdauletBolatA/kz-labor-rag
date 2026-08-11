"""Отметка вопросов отревьюированными.

Флаг ``reviewed_by_human`` — гейт запуска baseline, поэтому важно, чтобы его
нельзя было проставить случайно: ни несуществующему вопросу, ни черновому
слоту, ни молча при опечатке в id.
"""

from __future__ import annotations

import json

import pytest

from kz_labor_rag.eval.dataset import load_dataset
from kz_labor_rag.review_cli import main

BASE = {
    "question": "Меня уволили в отпуске, так можно?",
    "lang": "ru",
    "origin": "synthetic",
    "required_articles": ["54"],
    "evidence": "Не допускается расторжение трудового договора",
}


@pytest.fixture
def dataset_file(tmp_path):
    path = tmp_path / "ds.jsonl"
    rows = [
        {"id": "syn_001", **BASE},
        {"id": "syn_002", **BASE},
        {"id": "syn_003", **BASE, "reviewed_by_human": True},
        {"id": "real_001", "lang": "ru", "origin": "real", "status": "draft"},
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def run(dataset_file, *argv) -> int:
    return main([*argv, "--dataset", str(dataset_file)])


def flags(dataset_file) -> dict[str, bool]:
    return {q.id: q.reviewed_by_human for q in load_dataset(dataset_file)}


class TestStatus:
    def test_reports_pending_and_exits_nonzero(self, dataset_file, capsys):
        assert run(dataset_file, "status") == 1
        out = capsys.readouterr().out
        assert "готовых вопросов   : 3" in out
        assert "отревьюировано     : 1" in out
        assert "ждут ревью         : 2" in out
        assert "черновых слотов    : 1" in out

    def test_exits_zero_when_everything_reviewed(self, dataset_file, capsys):
        run(dataset_file, "mark", "--all")
        assert run(dataset_file, "status") == 0
        assert "Все готовые вопросы отревьюированы" in capsys.readouterr().out


class TestMark:
    def test_marks_named_questions_only(self, dataset_file):
        assert run(dataset_file, "mark", "syn_001") == 0
        assert flags(dataset_file) == {
            "real_001": False,
            "syn_001": True,
            "syn_002": False,
            "syn_003": True,
        }

    def test_mark_all_covers_ready_questions(self, dataset_file):
        assert run(dataset_file, "mark", "--all") == 0
        marks = flags(dataset_file)
        assert marks["syn_001"] and marks["syn_002"] and marks["syn_003"]
        # Черновой слот остаётся нетронутым.
        assert marks["real_001"] is False

    def test_unknown_id_is_rejected_and_changes_nothing(self, dataset_file, capsys):
        before = flags(dataset_file)
        assert run(dataset_file, "mark", "syn_001", "syn_999") == 1
        assert "syn_999" in capsys.readouterr().err
        # Опечатка не должна частично применяться.
        assert flags(dataset_file) == before

    def test_draft_slot_cannot_be_reviewed(self, dataset_file, capsys):
        assert run(dataset_file, "mark", "real_001") == 1
        assert "ревьюировать нечего" in capsys.readouterr().err
        assert flags(dataset_file)["real_001"] is False

    def test_no_ids_and_no_all_is_an_error(self, dataset_file, capsys):
        assert run(dataset_file, "mark") == 1
        assert "--all" in capsys.readouterr().err

    def test_reports_how_many_remain(self, dataset_file, capsys):
        run(dataset_file, "mark", "syn_001")
        assert "Осталось без ревью: 1" in capsys.readouterr().out

    def test_marking_is_idempotent(self, dataset_file):
        run(dataset_file, "mark", "syn_001")
        run(dataset_file, "mark", "syn_001")
        assert flags(dataset_file)["syn_001"] is True


class TestUnmark:
    def test_removes_the_flag(self, dataset_file):
        assert run(dataset_file, "unmark", "syn_003") == 0
        assert flags(dataset_file)["syn_003"] is False


class TestFileIntegrity:
    def test_other_fields_survive_rewrite(self, dataset_file):
        before = {q.id: q for q in load_dataset(dataset_file)}
        run(dataset_file, "mark", "syn_001")
        after = {q.id: q for q in load_dataset(dataset_file)}

        for qid, question in before.items():
            expected = question.to_dict()
            expected["reviewed_by_human"] = qid == "syn_001" or expected["reviewed_by_human"]
            assert after[qid].to_dict() == expected

    def test_file_stays_sorted_by_id(self, dataset_file):
        run(dataset_file, "mark", "syn_002")
        ids = [
            json.loads(line)["id"]
            for line in dataset_file.read_text(encoding="utf-8").splitlines()
        ]
        assert ids == sorted(ids)
