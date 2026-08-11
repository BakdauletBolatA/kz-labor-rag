"""Парсер ТК РК из сохранённого HTML ИПС «Әділет».

Разбирает документ в структуру раздел → глава → статья → пункт. Структура
нужна не сама по себе: она вход для structure-aware чанкинга, который позже
станет отдельной измеряемой итерацией, и для автосверки цитат в eval-датасете.

Три особенности разметки, из-за которых наивный парсер ломается:

1. **Заголовок статьи опознаётся по ``<p><b>Статья N. …</b></p>``, а не по
   якорю.** У 203 статей из 204 внутри ``<b>`` стоит ``<a name="zN">``, но у
   статьи 140 якорь перехвачен блоком «Примечание ИЗПИ!», стоящим перед
   заголовком. Зато перекрёстных ссылок вида ``<a href="#z…">`` в документе нет
   вовсе, поэтому упоминание «в соответствии со статьёй 52» в теле никогда не
   попадает в ``<b>`` и с заголовком не путается.

2. **Примечания об изменениях — не нормативный текст.** Они приходят как
   ``<p class="note">`` и ``<span class="note">`` («Сноска. Статья 140 с
   изменениями, внесенными Законом РК от …»). Выносятся в метаданные.

3. **Примечания ИЗПИ о будущих редакциях** размечены ``<font color="#FF0000">``
   и сообщают, что норма изменится с будущей даты. Соседство действующей и
   будущей формулировок — реальное свойство этого документа; они тоже выносятся
   в метаданные, чтобы в индекс не попал текст, ещё не вступивший в силу.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from selectolax.parser import HTMLParser

from kz_labor_rag.types import article_sort_key, normalize_article

PARSER_VERSION = "1.0"

# Контейнер с телом документа и элемент, с которого начинается обвязка сайта
# («Состояние базы», «Последние документы», подвал ИПС).
CONTENT_SELECTOR = "div.main"
CHROME_MARKER_CLASS = "aftertext"

BLOCK_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "div", "table", "ul"})

_ARTICLE_HEAD = re.compile(r"^Стать[яи]\s+(\d+(?:-\d+)?)\.\s*(.*)$", re.S)
_SECTION_HEAD = re.compile(r"^РАЗДЕЛ\s+(\d+)\.\s*(.*)$", re.I)
_CHAPTER_HEAD = re.compile(r"^Глава\s+(\d+(?:-\d+)?)\.\s*(.*)$", re.I)
_PART_HEAD = re.compile(r"^(ОБЩАЯ|ОСОБЕННАЯ)\s+ЧАСТЬ$", re.I)
# Начало пункта: «1.», «2-1.». Подпункты («1)») к пунктам не приравниваются:
# нумерация пунктов и подпунктов в кодексе пересекается, и смешивать их значит
# ломать preferred_clause в eval-датасете.
_CLAUSE_START = re.compile(r"^(\d+(?:-\d+)?)\.\s+(.*)$", re.S)
_EDITION_DATE = re.compile(r"Документы по состоянию на:?\s*([\d.]+)")
# «В заголовок статьи 140 предусматривается изменение…», «В статью 132 …».
_IZPI_TARGET = re.compile(r"стать[юия]\s+(\d+(?:-\d+)?)", re.I)
# Шапка блока примечаний ИЗПИ: информации не несёт, в метаданные не идёт.
_IZPI_HEADER = re.compile(r"^Примечание\s+ИЗПИ!?$", re.I)


def split_headings(text: str) -> list[str]:
    """Разбить блок заголовков на отдельные заголовки.

    В ``<h3>`` через ``<br>`` идут подряд «ОБЩАЯ ЧАСТЬ», «РАЗДЕЛ 1. …»,
    «Глава 1. …». Но тот же ``<br>`` встречается и как перенос строки внутри
    одного длинного названия главы. Считать каждую строку отдельным заголовком
    нельзя: «Глава 2. ГОСУДАРСТВЕННОЕ РЕГУЛИРОВАНИЕ В ОБЛАСТИ ТРУДОВЫХ /
    ОТНОШЕНИЙ» обрежется на полуслове. Поэтому новая строка начинает заголовок
    только если сама им выглядит, иначе приклеивается к предыдущему.
    """
    headings: list[str] = []
    for line in (squash(raw_line) for raw_line in text.split("\n")):
        if not line:
            continue
        starts_heading = bool(
            _PART_HEAD.match(line) or _SECTION_HEAD.match(line) or _CHAPTER_HEAD.match(line)
        )
        if starts_heading or not headings:
            headings.append(line)
        else:
            headings[-1] = f"{headings[-1]} {line}"
    return headings


class ParseError(RuntimeError):
    """HTML не похож на документ ИПС «Әділет»."""


def squash(text: str) -> str:
    """Схлопнуть пробелы, неразрывные в том числе."""
    return " ".join(text.replace("\xa0", " ").split())


@dataclass(frozen=True)
class Clause:
    """Пункт статьи."""

    number: str
    text: str


@dataclass(frozen=True)
class Article:
    """Статья кодекса вместе с местом в структуре документа."""

    number: str
    title: str
    part: str = ""
    section: str = ""
    chapter: str = ""
    clauses: tuple[Clause, ...] = ()
    text: str = ""
    amendment_notes: tuple[str, ...] = ()
    izpi_notes: tuple[str, ...] = ()

    @property
    def heading(self) -> str:
        return f"Статья {self.number}. {self.title}"

    @property
    def full_text(self) -> str:
        """Заголовок плюс текст. Именно это идёт в индексацию."""
        return f"{self.heading}\n{self.text}".strip()

    @property
    def clause_numbers(self) -> tuple[str, ...]:
        return tuple(c.number for c in self.clauses)

    @property
    def has_future_edition(self) -> bool:
        """Для статьи объявлена редакция, вступающая в силу позже.

        Такие статьи требуют внимания при разметке эталона: «правильная»
        формулировка зависит от даты.
        """
        return bool(self.izpi_notes)

    @property
    def is_repealed(self) -> bool:
        """Статья исключена из кодекса и нормативного текста не имеет.

        У таких статей остаётся только сноска «Статья N исключена Законом
        РК от …». Их обязательно отличать от статей с непустым текстом: пустая
        статья, попавшая в ``required_articles``, делает вопрос гарантированно
        непроходимым, а выглядит это как провал поиска.
        """
        if self.text.strip():
            return False
        return any(
            marker in note.lower()
            for note in self.amendment_notes
            for marker in ("исключена", "утратила силу")
        )


@dataclass(frozen=True)
class LaborCode:
    """Разобранный кодекс."""

    articles: tuple[Article, ...]
    edition_date: str | None = None
    parser_version: str = PARSER_VERSION
    meta: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.articles)

    def __iter__(self):
        return iter(self.articles)

    @property
    def by_number(self) -> dict[str, Article]:
        return {a.number: a for a in self.articles}

    def article_texts(self) -> dict[str, str]:
        """Вход для ``validate_against_corpus``: номер статьи → её текст."""
        return {a.number: a.full_text for a in self.articles}

    @property
    def stats(self) -> dict[str, int]:
        return {
            "articles": len(self.articles),
            "clauses": sum(len(a.clauses) for a in self.articles),
            "sections": len({a.section for a in self.articles if a.section}),
            "chapters": len({a.chapter for a in self.articles if a.chapter}),
            "with_amendment_notes": sum(1 for a in self.articles if a.amendment_notes),
            "with_future_edition": sum(1 for a in self.articles if a.has_future_edition),
            "repealed": sum(1 for a in self.articles if a.is_repealed),
            "empty_not_repealed": sum(
                1 for a in self.articles if not a.text.strip() and not a.is_repealed
            ),
        }

    @property
    def in_force(self) -> tuple[Article, ...]:
        """Только действующие статьи — то, что имеет смысл индексировать."""
        return tuple(a for a in self.articles if not a.is_repealed)


def _is_note(node) -> bool:
    """Примечание об изменениях: ``class="note"`` на самом узле или внутри."""
    classes = node.attributes.get("class") or ""
    return "note" in classes.split()


def _is_izpi(node) -> bool:
    """Примечание ИЗПИ о будущей редакции: красный шрифт."""
    if node.tag == "font" and (node.attributes.get("color") or "").upper() == "#FF0000":
        return True
    return node.css_first('font[color="#FF0000"]') is not None and "Примечание ИЗПИ" in node.text()


def _visible_text(node, *, drop_notes: bool) -> str:
    """Текст узла без служебных вставок.

    Примечания вырезаются до извлечения текста, иначе «Сноска. Статья 54 с
    изменениями…» приклеится к последнему пункту статьи и попадёт в чанк.
    """
    if not drop_notes:
        return squash(node.text())

    clone = HTMLParser(node.html or "")
    for selector in (".note", 'font[color="#FF0000"]'):
        for junk in clone.css(selector):
            # Именно замена на пробел, а не decompose: вырезание узла склеивает
            # соседние текстовые куски, и «пунктом 1-1<note/>статьи 52»
            # превращается в «1-1статьи». Дословные цитаты в eval-датасете
            # после такой склейки перестают находиться в тексте.
            junk.replace_with(" ")
    return squash(clone.text())


def _split_clauses(paragraphs: list[str]) -> tuple[Clause, ...]:
    """Разложить абзацы статьи по пунктам.

    Абзац, начинающийся с «N.» — новый пункт. Всё остальное (подпункты «N)»,
    продолжения, цитаты) приклеивается к текущему пункту. Текст до первого
    пронумерованного абзаца попадает в пункт с пустым номером: у части статей
    кодекса нумерации пунктов нет вовсе, и терять их текст нельзя.
    """
    clauses: list[tuple[str, list[str]]] = []
    for para in paragraphs:
        if not para:
            continue
        if m := _CLAUSE_START.match(para):
            clauses.append((m.group(1), [m.group(2).strip()]))
        elif clauses:
            clauses[-1][1].append(para)
        else:
            clauses.append(("", [para]))
    return tuple(Clause(number=num, text=" ".join(parts).strip()) for num, parts in clauses)


def parse_labor_code(html: str, *, strip_amendment_notes: bool = True) -> LaborCode:
    """Разобрать сохранённый HTML в структуру кодекса."""
    # <br> несёт границу строки: без замены «ТРУДОВЫХ<br>ОТНОШЕНИЙ»
    # склеивается в «ТРУДОВЫХОТНОШЕНИЙ».
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    tree = HTMLParser(html)

    container = tree.css_first(CONTENT_SELECTOR)
    if container is None:
        raise ParseError(
            f"не найден контейнер документа ({CONTENT_SELECTOR}). "
            "Похоже, разметка ИПС «Әділет» изменилась — парсер нужно чинить, "
            "а не запускать на том, что получилось."
        )

    edition = _EDITION_DATE.search(squash(tree.body.text()))

    part = section = chapter = ""
    articles: list[Article] = []
    current: dict | None = None
    # Примечания ИЗПИ о будущих редакциях стоят ПЕРЕД заголовком статьи,
    # к которой относятся, поэтому «текущая статья» для них — неправильный
    # адресат. Привязка двухступенчатая: если примечание само называет статью
    # («В заголовок статьи 140 предусматривается изменение…») — берём её;
    # если нет («Подпункт 2) предусматривается в редакции…») — примечание
    # относится к статье, заголовок которой идёт следующим.
    izpi_notes: list[tuple[str, str | None]] = []
    pending_izpi: list[str] = []

    def buffer_izpi(node) -> None:
        text = squash(node.text())
        if not text or _IZPI_HEADER.match(text):
            return
        # Примечание, встреченное посреди тела статьи, относится к ней самой.
        # Примечание до первого абзаца — к статье, заголовок которой впереди.
        if current is not None and current["paragraphs"]:
            izpi_notes.append((text, current["number"]))
        elif text not in pending_izpi:
            pending_izpi.append(text)

    def flush() -> None:
        if current is None:
            return
        clauses = _split_clauses(current["paragraphs"])
        articles.append(
            Article(
                number=current["number"],
                title=current["title"],
                part=current["part"],
                section=current["section"],
                chapter=current["chapter"],
                clauses=clauses,
                text="\n".join(c.text for c in clauses).strip(),
                amendment_notes=tuple(current["notes"]),
            )
        )

    def apply_structural_heading(line: str) -> bool:
        """Обновить текущее место в структуре. True, если строка — заголовок."""
        nonlocal part, section, chapter
        if _PART_HEAD.match(line):
            part, section, chapter = line, "", ""
        elif _SECTION_HEAD.match(line):
            section, chapter = line, ""
        elif _CHAPTER_HEAD.match(line):
            chapter = line
        else:
            return False
        return True

    for node in container.traverse(include_text=False):
        # Обвязка сайта: всё, что после этого элемента, к кодексу не относится.
        if CHROME_MARKER_CLASS in (node.attributes.get("class") or "").split():
            break

        # Сноски приходят и как <p class="note">, и как голый <span class="note">
        # между абзацами. Второй случай в BLOCK_TAGS не входит, но потерять его
        # нельзя: это история правок статьи.
        if node.tag == "span" and _is_note(node) and current is not None:
            if (text := squash(node.text())) and text not in current["notes"]:
                current["notes"].append(text)
            continue

        if node.tag not in BLOCK_TAGS and node.tag != "font":
            continue

        if node.tag == "h3":
            for line in split_headings(node.text()):
                apply_structural_heading(line)
            continue

        # Примечания ИЗПИ приходят голым <font color="#FF0000"> между абзацами.
        if node.tag == "font" and _is_izpi(node):
            buffer_izpi(node)
            continue

        if node.tag != "p":
            continue

        bold = node.css_first("b")

        # Глава 18 размечена <p>, а не <h3>, в отличие от остальных 22 глав.
        # Опираться только на <h3> — значит потерять её и приписать её статьям
        # предыдущую главу.
        if apply_structural_heading(squash((bold or node).text())):
            continue

        if bold is not None and (m := _ARTICLE_HEAD.match(squash(bold.text()))):
            flush()
            number = normalize_article(m.group(1))
            # Всё, что накопилось до заголовка, относится к этой статье.
            izpi_notes.extend((note, number) for note in pending_izpi)
            pending_izpi.clear()
            current = {
                "number": number,
                "title": m.group(2).strip(),
                "part": part,
                "section": section,
                "chapter": chapter,
                "paragraphs": [],
                "notes": [],
            }
            continue

        if current is None:
            continue

        if _is_note(node):
            if text := squash(node.text()):
                current["notes"].append(text)
            continue

        if _is_izpi(node):
            buffer_izpi(node)
            continue

        if text := _visible_text(node, drop_notes=strip_amendment_notes):
            current["paragraphs"].append(text)

    flush()
    # Примечания, для которых заголовок статьи так и не встретился: последний
    # шанс — номер статьи, названный в самом тексте примечания.
    izpi_notes.extend((note, None) for note in pending_izpi)

    if not articles:
        raise ParseError("в документе не найдено ни одной статьи")

    # Привязать примечания ИЗПИ к статьям, которые в них названы.
    by_number = {a.number: a for a in articles}
    attached: dict[str, list[str]] = {}
    unattached: list[str] = []
    for note, fallback in izpi_notes:
        target = None
        if (m := _IZPI_TARGET.search(note)) and normalize_article(m.group(1)) in by_number:
            target = normalize_article(m.group(1))
        elif fallback in by_number:
            target = fallback
        if target is None:
            unattached.append(note)
        elif note not in attached.setdefault(target, []):
            attached[target].append(note)

    articles = [
        (a if a.number not in attached else replace(a, izpi_notes=tuple(attached[a.number])))
        for a in articles
    ]

    return LaborCode(
        articles=tuple(articles),
        edition_date=edition.group(1) if edition else None,
        meta={
            "strip_amendment_notes": str(strip_amendment_notes),
            "izpi_notes_total": str(len(izpi_notes)),
            # Примечание, из текста которого не удалось понять, к какой статье
            # оно относится. Молча терять их нельзя: каждое такое примечание —
            # норма, формулировка которой зависит от даты.
            "izpi_notes_unattached": str(len(unattached)),
        },
    )


def parse_file(path: str | Path, *, strip_amendment_notes: bool = True) -> LaborCode:
    return parse_labor_code(
        Path(path).read_text(encoding="utf-8", errors="replace"),
        strip_amendment_notes=strip_amendment_notes,
    )
