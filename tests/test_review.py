"""Скрипт ревью: каждое действие сразу попадает в файл и ничего лишнего не трогает."""

from __future__ import annotations

import json

import pytest

from kz_labor_rag.corpus.parser import parse_labor_code
from kz_labor_rag.eval.dataset import load_dataset
from kz_labor_rag.eval.review import ReviewSession, clause_quote, parse_clause_refs
from kz_labor_rag.types import ClauseRef

ARTICLES = """
<p><b>Статья 62. Выдача документов</b></p>
<p>&nbsp; 1. По требованию работника работодатель обязан в течение пяти
рабочих дней выдать справку.</p>
<p>&nbsp; 2. Работодатель обязан выдать документы в день увольнения работника.</p>
<p><b>Статья 83. Продолжительность междусменного отдыха</b></p>
<p>&nbsp; Отдых между сменами не может быть менее двенадцати часов.</p>
"""


@pytest.fixture
def code():
    return parse_labor_code(
        f'<html><body><div class="main">{ARTICLES}'
        '<div class="container_omega aftertext">x</div></div></body></html>'
    )


def row(qid: str, **overrides) -> dict:
    return {
        "id": qid,
        "question": f"вопрос {qid}",
        "lang": "ru",
        "origin": "synthetic",
        "type": "number",
        "required_articles": ["62"],
        "required_clauses": [{"article": "62", "clause": "1"}],
        "evidence": [{"article": "62", "quote": "в течение пяти рабочих дней"}],
        "verified": False,
        **overrides,
    }


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "questions.jsonl"
    rows = [
        row("a"),
        row("b"),
        {
            "id": "u",
            "question": "Какая ставка ИПН?",
            "lang": "ru",
            "origin": "synthetic",
            "type": "unanswerable",
            "notes": "Налоговый кодекс",
        },
    ]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return p


def session(path, code, answers):
    replies = iter(answers)
    out: list[str] = []
    return ReviewSession(path, code, ask=lambda _: next(replies), say=out.append), out


def by_id(path):
    return {q.id: q for q in load_dataset(path)}


class TestActions:
    def test_verify_marks_and_saves_immediately(self, path, code):
        s, _ = session(path, code, ["v", "q"])
        s.run()
        saved = by_id(path)
        assert saved["a"].verified is True
        assert saved["b"].verified is False

    def test_skip_changes_nothing(self, path, code):
        before = path.read_text("utf-8")
        s, _ = session(path, code, ["s", "s", "s"])
        s.run()
        assert path.read_text("utf-8") == before

    def test_delete_requires_confirmation(self, path, code):
        s, _ = session(path, code, ["d", "n", "d", "y", "q"])
        s.run()
        assert "a" not in by_id(path)
        assert "b" in by_id(path)

    def test_edit_replaces_clauses_and_derives_the_rest(self, path, code):
        s, _ = session(path, code, ["e", "62/2 83/", "v", "q"])
        s.run()
        q = by_id(path)["a"]
        assert q.required_clauses == (ClauseRef("62", "2"), ClauseRef("83", ""))
        assert q.required_articles == ("62", "83")
        assert q.type == "multi"
        assert q.verified is True
        assert [e.quote for e in q.evidence] == [
            "Работодатель обязан выдать документы в день увольнения работника.",
            "Отдых между сменами не может быть менее двенадцати часов.",
        ]

    def test_edit_to_one_clause_asks_for_type_when_it_was_multi(self, path, code):
        path.write_text(
            json.dumps(
                row(
                    "m",
                    type="multi",
                    required_articles=["62", "83"],
                    required_clauses=[
                        {"article": "62", "clause": "1"},
                        {"article": "83", "clause": ""},
                    ],
                    evidence=[
                        {"article": "62", "quote": "в течение пяти рабочих дней"},
                        {"article": "83", "quote": "не может быть менее двенадцати часов"},
                    ],
                ),
                ensure_ascii=False,
            )
            + "\n",
            "utf-8",
        )
        s, _ = session(path, code, ["e", "83/", "opinion", "number", "v"])
        s.run()
        assert by_id(path)["m"].type == "number"

    def test_bad_clause_is_reported_and_nothing_is_saved(self, path, code):
        before = path.read_text("utf-8")
        s, out = session(path, code, ["e", "62/9", "q"])
        s.run()
        assert path.read_text("utf-8") == before
        assert any("нет пункта '9'" in line for line in out)

    def test_quit_keeps_earlier_marks(self, path, code):
        s, _ = session(path, code, ["v", "q"])
        s.run()
        s2, _ = session(path, code, ["q"])
        s2.run()
        assert by_id(path)["a"].verified is True

    def test_unanswerable_shows_where_the_answer_is(self, path, code):
        s, out = session(path, code, ["v"])
        s.run(["u"])
        assert any("Налоговый кодекс" in line for line in out)
        assert by_id(path)["u"].verified is True

    def test_shows_full_clause_text(self, path, code):
        s, out = session(path, code, ["q"])
        s.run(["a"])
        assert any("выдать справку" in line for line in out)


class TestHelpers:
    def test_parse_refs(self):
        assert parse_clause_refs("52/1, 83/") == [ClauseRef("52", "1"), ClauseRef("83", "")]

    def test_parse_rejects_bare_article(self):
        with pytest.raises(ValueError, match="статья/пункт"):
            parse_clause_refs("52")

    def test_long_clause_is_cut_at_a_word(self):
        quote = clause_quote("слово " * 200)
        assert len(quote) <= 600
        assert not quote.endswith(" ")
