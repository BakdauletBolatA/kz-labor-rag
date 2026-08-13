"""Команда `kzrag-corpus quote`.

Ею размечают реальные и казахские вопросы: их пишут в датасет руками, и без
этой команды цитату пришлось бы перенабирать из кодекса. Перенабранная цитата
отличается от текста статьи одним символом — и обоснование разметки, которое
человек читает при ревью, перестаёт быть обоснованием.
"""

from __future__ import annotations

import json

import pytest

from kz_labor_rag.corpus.cli import main
from kz_labor_rag.corpus.evidence import extract_evidence
from kz_labor_rag.corpus.parser import parse_labor_code

ARTICLES = """
<h3>РАЗДЕЛ 2. ТРУДОВЫЕ ОТНОШЕНИЯ<br>Глава 4. ТРУДОВОЙ ДОГОВОР</h3>
<p><b>Статья 62. Выдача документов</b></p>
<p>&nbsp; 1. Работодатель обязан в течение пяти рабочих дней выдать справку.</p>
<p>&nbsp; 2. Справка выдается по требованию работника.</p>
<p><b>Статья 76. Работа в ночное время</b></p>
<p>&nbsp; 2. К работе не допускаются: несовершеннолетние; беременные женщины.</p>
<p><b>Статья 117. Профессиональные стандарты</b></p>
<p class="note">Сноска. Статья 117 исключена Законом РК от 04.07.2023.</p>
"""


@pytest.fixture
def raw(tmp_path):
    path = tmp_path / "code.html"
    path.write_text(
        f'<html><body><div class="main">{ARTICLES}'
        '<div class="container_omega aftertext">x</div></div></body></html>',
        encoding="utf-8",
    )
    return str(path)


def run(raw, *args) -> int:
    return main(["--raw", raw, *args])


class TestQuote:
    def test_prints_fragment_and_evidence_json(self, raw, capsys):
        assert run(raw, "quote", "62", "в течение пяти") == 0
        out = capsys.readouterr().out

        assert "Статья 62. Выдача документов" in out
        assert "пункты: 1, 2" in out

        payload = json.loads(out.strip().splitlines()[-1])
        assert payload["article"] == "62"
        assert "в течение пяти рабочих дней" in payload["quote"]

    def test_quote_matches_what_the_build_would_cut(self, raw, capsys):
        # Правило вырезания одно на всех. Разойдись оно — человек размечал бы
        # по одной цитате, а в датасет попадала бы другая.
        assert run(raw, "quote", "62", "в течение пяти") == 0
        printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["quote"]

        article = parse_labor_code(
            f'<html><body><div class="main">{ARTICLES}'
            '<div class="container_omega aftertext">x</div></div></body></html>'
        ).by_number["62"]
        assert printed == extract_evidence(article.full_text, "в течение пяти")

    def test_quote_is_verbatim(self, raw, capsys):
        assert run(raw, "quote", "76", "К работе не допускаются") == 0
        quote = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["quote"]
        # Точка с запятой не обрывает перечень — тот самый дефект ревью.
        assert "беременные женщины" in quote

    def test_unknown_article_fails(self, raw, capsys):
        assert run(raw, "quote", "999", "что угодно") == 1
        assert "не найдена" in capsys.readouterr().err

    def test_missing_anchor_points_at_show(self, raw, capsys):
        assert run(raw, "quote", "62", "такого текста в статье нет") == 1
        err = capsys.readouterr().err
        assert "Якорь не найден" in err
        assert "kzrag-corpus show 62" in err

    def test_repealed_article_is_refused(self, raw, capsys):
        # Исключённая статья непроходима в принципе: в required_articles она
        # выглядит как провал поиска, а не как ошибка разметки.
        assert run(raw, "quote", "117", "что угодно") == 1
        assert "исключена" in capsys.readouterr().err
