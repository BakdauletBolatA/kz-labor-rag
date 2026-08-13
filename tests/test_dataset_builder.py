"""Сборщик датасета.

Скрипт пересобирает синтетическую русскую часть и обязан сохранить всё
остальное. Раньше он этого не делал: слоты ``real_*`` пересоздавались пустыми
при каждом запуске, и первая же вписанная в файл реальная формулировка
исчезала на следующем ``make dataset`` вместе со ссылкой на источник.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kz_labor_rag.corpus.parser import parse_labor_code
from kz_labor_rag.eval.dataset import EvalDataset, EvalQuestion


def _load_builder():
    """Скрипт лежит в scripts/ и пакетом не является."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_synthetic_dataset.py"
    spec = importlib.util.spec_from_file_location("build_synthetic_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


REAL_ROW = {
    "id": "real_001",
    "question": "Могут ли не отдать трудовую книжку, пока я не подпишу обходной лист?",
    "origin": "real",
    "source_url": "https://example.kz/thread/1",
    "required_articles": ["62"],
    "acceptable_articles": [],
    "preferred_clause": None,
    "tags": ["увольнение"],
    "lang": "ru",
    "evidence": [{"article": "62", "quote": "работодатель обязан в течение пяти рабочих дней"}],
    "status": "ready",
    "reviewed_by_human": True,
}

KK_ROW = {
    "id": "kk_001",
    "question": "Жыл сайынғы демалыс қанша күн?",
    "origin": "synthetic",
    "source_url": None,
    "required_articles": ["88"],
    "acceptable_articles": [],
    "preferred_clause": None,
    "tags": ["отпуск"],
    "lang": "kk",
    "evidence": [{"article": "88", "quote": "двадцать четыре календарных дня"}],
    "status": "ready",
    "reviewed_by_human": False,
}

SYNTHETIC_ROW = {
    "id": "syn_001",
    "question": "Меня хотят уволить, а я на больничном. Так можно?",
    "origin": "synthetic",
    "source_url": None,
    "required_articles": ["54"],
    "acceptable_articles": [],
    "preferred_clause": None,
    "tags": ["увольнение"],
    "lang": "ru",
    "evidence": [{"article": "54", "quote": "Не допускается расторжение трудового договора"}],
    "status": "ready",
    "reviewed_by_human": True,
}

EMPTY_SLOT = {"id": "real_002", "lang": "ru", "origin": "real", "status": "draft"}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


class TestOwnership:
    def test_synthetic_russian_belongs_to_the_script(self):
        assert builder.is_generated(SYNTHETIC_ROW)

    def test_real_question_is_authored_by_hand(self):
        assert not builder.is_generated(REAL_ROW)

    def test_kazakh_question_is_authored_by_hand(self):
        # Казахский вопрос тоже origin='synthetic', и различает их только язык.
        assert not builder.is_generated(KK_ROW)

    def test_empty_slot_is_authored(self):
        # Иначе слот пересоздавался бы, даже когда он уже заполнен.
        assert not builder.is_generated(EMPTY_SLOT)


class TestCarryOverAuthored:
    def test_real_question_survives_rebuild(self, tmp_path):
        previous = write_jsonl(tmp_path / "d.jsonl", [SYNTHETIC_ROW, REAL_ROW])
        authored, problems = builder.carry_over_authored(previous)
        assert problems == []
        assert [q.id for q in authored] == ["real_001"]
        assert authored[0].source_url == "https://example.kz/thread/1"
        assert authored[0].reviewed_by_human is True

    def test_kazakh_question_survives_rebuild(self, tmp_path):
        previous = write_jsonl(tmp_path / "d.jsonl", [SYNTHETIC_ROW, KK_ROW])
        authored, problems = builder.carry_over_authored(previous)
        assert [q.id for q in authored] == ["kk_001"]
        assert authored[0].lang == "kk"

    def test_generated_rows_are_dropped(self, tmp_path):
        # Источник истины для синтетики — SPECS. Удалённая спецификация обязана
        # исчезнуть из датасета, а не остаться сиротой в файле.
        previous = write_jsonl(tmp_path / "d.jsonl", [SYNTHETIC_ROW])
        authored, problems = builder.carry_over_authored(previous)
        assert authored == []
        assert problems == []

    def test_broken_authored_row_is_reported_not_silently_dropped(self, tmp_path):
        broken = {**REAL_ROW, "evidence": []}
        previous = write_jsonl(tmp_path / "d.jsonl", [broken])
        authored, problems = builder.carry_over_authored(previous)
        assert authored == []
        assert len(problems) == 1
        assert "real_001" in problems[0]

    def test_missing_file_is_not_an_error(self, tmp_path):
        authored, problems = builder.carry_over_authored(tmp_path / "нет.jsonl")
        assert (authored, problems) == ([], [])

    def test_empty_slots_are_carried_as_drafts(self, tmp_path):
        previous = write_jsonl(tmp_path / "d.jsonl", [EMPTY_SLOT])
        authored, problems = builder.carry_over_authored(previous)
        assert problems == []
        assert [q.id for q in authored] == ["real_002"]
        assert authored[0].is_draft


class TestReserveSlots:
    def test_all_slots_created_on_empty_dataset(self):
        slots = builder.reserve_slots(set())
        assert [q.id for q in slots] == [f"real_{i:03d}" for i in range(1, 16)]
        assert all(q.is_draft and q.origin == "real" for q in slots)

    def test_filled_question_is_not_overwritten_by_a_slot(self):
        # Тот самый дефект: real_001 заполнен, а сборка подкладывала на его
        # место пустой черновик.
        slots = builder.reserve_slots({"real_001"})
        assert "real_001" not in {q.id for q in slots}
        assert len(slots) == 14

    def test_nothing_reserved_when_every_slot_is_taken(self):
        taken = {f"real_{i:03d}" for i in range(1, 16)}
        assert builder.reserve_slots(taken) == []


ARTICLES = """
<h3>РАЗДЕЛ 2. ТРУДОВЫЕ ОТНОШЕНИЯ<br>Глава 4. ТРУДОВОЙ ДОГОВОР</h3>
<p><b>Статья 62. Выдача документов</b></p>
<p>&nbsp; 1. По требованию работника работодатель обязан в течение пяти
рабочих дней выдать справку.</p>
<p><b>Статья 88. Продолжительность отпуска</b></p>
<p>&nbsp; Отпуск предоставляется продолжительностью двадцать четыре календарных дня.</p>
"""


@pytest.fixture
def code():
    return parse_labor_code(
        f'<html><body><div class="main">{ARTICLES}'
        '<div class="container_omega aftertext">x</div></div></body></html>'
    )


class TestFullRebuild:
    """Сборка целиком, на фальшивом корпусе из двух статей.

    Юнит-тесты выше проверяют части по отдельности; здесь проверяется то, из-за
    чего всё затевалось: пересборка не должна уносить рукописные вопросы.
    """

    @pytest.fixture
    def build(self, tmp_path, monkeypatch):
        raw = tmp_path / "code.html"
        raw.write_text(
            f'<html><body><div class="main">{ARTICLES}'
            '<div class="container_omega aftertext">x</div></div></body></html>',
            encoding="utf-8",
        )
        out = tmp_path / "dataset.jsonl"
        monkeypatch.setattr(builder, "RAW", str(raw))
        monkeypatch.setattr(builder, "OUT", str(out))
        monkeypatch.setattr(builder, "REVIEW", str(tmp_path / "REVIEW.md"))
        monkeypatch.setattr(
            builder,
            "SPECS",
            [
                (
                    "syn_001",
                    "Сколько ждать справку с работы после увольнения?",
                    ["62"],
                    [],
                    ("62", "1"),
                    ["увольнение"],
                    ("62", "в течение пяти"),
                    "",
                )
            ],
        )
        return out

    def read(self, path):
        return {
            json.loads(line)["id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def test_first_build_creates_all_slots(self, build):
        assert builder.main() == 0
        rows = self.read(build)
        assert rows["syn_001"]["status"] == "ready"
        assert [f"real_{i:03d}" for i in range(1, 16)] == sorted(
            r for r in rows if r.startswith("real_")
        )

    def test_authored_questions_survive_a_rebuild(self, build):
        assert builder.main() == 0
        rows = self.read(build)
        rows["real_001"] = REAL_ROW
        rows["kk_001"] = KK_ROW
        write_jsonl(build, list(rows.values()))

        assert builder.main() == 0

        after = self.read(build)
        assert after["real_001"]["question"] == REAL_ROW["question"]
        assert after["real_001"]["source_url"] == REAL_ROW["source_url"]
        assert after["real_001"]["reviewed_by_human"] is True
        assert after["kk_001"]["lang"] == "kk"
        # Слот под уже заполненный real_001 не пересоздан, остальные на месте.
        assert after["real_002"]["status"] == "draft"
        assert len([r for r in after.values() if r["status"] == "draft"]) == 14

    def test_rebuild_is_idempotent(self, build):
        assert builder.main() == 0
        rows = self.read(build)
        rows["real_001"] = REAL_ROW
        write_jsonl(build, list(rows.values()))

        assert builder.main() == 0
        once = build.read_text(encoding="utf-8")
        assert builder.main() == 0
        assert build.read_text(encoding="utf-8") == once

    def test_fabricated_quote_in_an_authored_question_stops_the_build(self, build, capsys):
        assert builder.main() == 0
        rows = self.read(build)
        rows["real_001"] = {
            **REAL_ROW,
            "evidence": [{"article": "62", "quote": "работодатель обязан выдать справку за час"}],
        }
        write_jsonl(build, list(rows.values()))

        # Рукописная цитата проходит ту же сверку с корпусом, что и вырезанная
        # скриптом. Иначе руками можно было бы вписать любое обоснование.
        assert builder.main() == 1
        assert "real_001" in capsys.readouterr().err

    def test_broken_authored_row_stops_the_build(self, build, capsys):
        assert builder.main() == 0
        rows = self.read(build)
        rows["real_001"] = {**REAL_ROW, "required_articles": [], "evidence": []}
        write_jsonl(build, list(rows.values()))

        assert builder.main() == 1
        assert "real_001" in capsys.readouterr().err


class TestWriteReview:
    """Файл ревью — единственное, по чему человек проверяет разметку.

    Вопрос, не попавший в него, не будет отревьюирован, а гейт его требует.
    """

    @pytest.fixture(autouse=True)
    def _in_tmp_dir(self, tmp_path, monkeypatch):
        # write_review пишет в модульную константу REVIEW.
        monkeypatch.setattr(builder, "REVIEW", str(tmp_path / "REVIEW.md"))
        self.review = tmp_path / "REVIEW.md"

    def test_survives_a_dataset_without_drafts(self, code):
        # Когда все слоты заполнены, черновиков не остаётся. Раньше это был
        # IndexError по drafts[0] — сборка падала ровно в тот момент, ради
        # которого затевалась.
        dataset = EvalDataset(questions=(EvalQuestion.from_dict(REAL_ROW),))
        builder.write_review(dataset, code)
        assert "Незаполненных слотов не осталось." in self.review.read_text(encoding="utf-8")

    def test_untagged_question_is_not_lost(self, code):
        dataset = EvalDataset(questions=(EvalQuestion.from_dict({**REAL_ROW, "tags": []}),))
        builder.write_review(dataset, code)
        text = self.review.read_text(encoding="utf-8")
        assert "## без темы" in text
        assert "real_001" in text

    def test_question_with_two_tags_appears_once(self, code):
        row = {**REAL_ROW, "tags": ["увольнение", "оплата"]}
        dataset = EvalDataset(questions=(EvalQuestion.from_dict(row),))
        builder.write_review(dataset, code)
        assert self.review.read_text(encoding="utf-8").count("### real_001") == 1

    def test_authored_question_shows_slice_and_source(self, code):
        dataset = EvalDataset(questions=(EvalQuestion.from_dict(REAL_ROW),))
        builder.write_review(dataset, code)
        text = self.review.read_text(encoding="utf-8")
        assert "**Срез:** ru, real — https://example.kz/thread/1" in text

    def test_synthetic_question_has_no_slice_line(self, code):
        dataset = EvalDataset(questions=(EvalQuestion.from_dict(KK_ROW),))
        builder.write_review(dataset, code)
        assert "**Срез:** kk, synthetic" in self.review.read_text(encoding="utf-8")
