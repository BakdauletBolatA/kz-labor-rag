"""Схема и загрузка eval-датасета.

Датасет — JSONL, по одному вопросу на строку, в git. Он появляется до baseline
и до какого-либо RAG: разметка, сделанная после того, как увидели выдачу,
подгоняет метрику под систему, а не измеряет её.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kz_labor_rag.types import ClauseRef, article_sort_key, normalize_article, normalize_clause

Origin = Literal["real", "synthetic"]
Lang = Literal["ru", "kk"]
Status = Literal["ready", "draft"]

DATASET_SCHEMA_VERSION = "2.0"


class DatasetError(ValueError):
    """Датасет нарушает схему. Всегда фатально: молча чинить эталон нельзя."""


@dataclass(frozen=True)
class EvidenceQuote:
    """Дословная цитата из конкретной статьи, обосновывающая разметку."""

    article: str
    quote: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "article", normalize_article(self.article))
        object.__setattr__(self, "quote", self.quote.strip())


@dataclass(frozen=True)
class EvalQuestion:
    """Один размеченный вопрос.

    ``evidence`` — список дословных цитат, по одной на каждую обязательную
    статью. Существует не для метрик, а для ревью человеком: по цитатам видно,
    из каких норм взят эталон, не открывая кодекс.

    Список, а не одна строка, потому что вопрос законно может опираться на
    несколько статей: «сколько дней декрета и сохранят ли должность» — это
    ст. 99 и ст. 100, и одна цитата вторую половину ответа не доказывает.
    Одна цитата на многосоставный вопрос молча скрывала неполноту разметки,
    поэтому схема требует цитату к КАЖДОЙ обязательной статье.

    ``source_url`` заполняется только у реальных вопросов — ссылка на тред,
    откуда взята формулировка. У синтетических всегда ``None``.

    ``status='draft'`` — незаполненный слот: сам вопрос ещё не написан или не
    размечен. Черновики лежат в том же файле, что и готовые вопросы, чтобы
    было видно, сколько ещё осталось, но не участвуют ни в метриках, ни в
    подсчёте укомплектованности. Требования к обязательным полям к ним не
    применяются — иначе пустой слот невозможно было бы записать.
    """

    id: str
    question: str
    lang: Lang
    origin: Origin
    required_articles: tuple[str, ...]
    evidence: tuple[EvidenceQuote, ...]
    acceptable_articles: tuple[str, ...] = ()
    preferred_clause: ClauseRef | None = None
    source_url: str | None = None
    status: Status = "ready"
    reviewed_by_human: bool = False
    notes: str = ""
    tags: tuple[str, ...] = ()

    @property
    def is_draft(self) -> bool:
        return self.status == "draft"

    @classmethod
    def from_dict(cls, raw: dict, *, source: str = "<dict>") -> EvalQuestion:
        def fail(msg: str) -> None:
            raise DatasetError(f"{source}: вопрос {raw.get('id', '<без id>')}: {msg}")

        status = raw.get("status", "ready")
        if status not in ("ready", "draft"):
            fail(f"status='{status}', допустимы только 'ready' и 'draft'")

        if not raw.get("id"):
            fail("обязательное поле 'id' пусто или отсутствует")

        if status == "ready":
            for key in ("question", "lang", "origin", "required_articles", "evidence"):
                if not raw.get(key):
                    fail(f"обязательное поле '{key}' пусто или отсутствует")
        elif not raw.get("lang") or not raw.get("origin"):
            fail("даже у черновика должны быть заполнены 'lang' и 'origin'")

        if raw["lang"] not in ("ru", "kk"):
            fail(f"lang='{raw['lang']}', допустимы только 'ru' и 'kk'")
        if raw["origin"] not in ("real", "synthetic"):
            fail(f"origin='{raw['origin']}', допустимы только 'real' и 'synthetic'")

        # Номер статьи — строка: в кодексе есть статьи с составными номерами
        # («73-1», «126-1»). Числа в JSON принимаются и приводятся к строке,
        # чтобы разметка не ломалась из-за того, что кто-то написал 54, а не "54".
        required = tuple(normalize_article(a) for a in raw.get("required_articles") or ())
        acceptable = tuple(normalize_article(a) for a in raw.get("acceptable_articles") or ())
        overlap = set(required) & set(acceptable)
        if overlap:
            fail(
                f"статьи {sorted(overlap, key=article_sort_key)} перечислены "
                "и в required, и в acceptable — "
                "эталон должен быть однозначным"
            )

        evidence = tuple(
            EvidenceQuote(article=e["article"], quote=e["quote"])
            for e in (raw.get("evidence") or ())
        )
        if status == "ready":
            covered = {e.article for e in evidence}
            if missing := [a for a in required if a not in covered]:
                fail(
                    f"нет цитаты к обязательным статьям {sorted(missing, key=article_sort_key)}. "
                    "Каждая обязательная статья должна быть обоснована своей цитатой, "
                    "иначе неполнота разметки не видна при ревью"
                )
            if stray := [a for a in covered if a not in required]:
                fail(
                    f"цитаты взяты из статей {sorted(stray, key=article_sort_key)}, "
                    "которых нет в required_articles"
                )

        preferred = None
        if pc := raw.get("preferred_clause"):
            preferred = ClauseRef(
                article=normalize_article(pc["article"]), clause=str(pc["clause"])
            )
            if preferred.article not in required:
                fail(
                    f"preferred_clause указывает на статью {preferred.article}, "
                    "которой нет в required_articles"
                )

        return cls(
            id=str(raw["id"]),
            question=str(raw.get("question") or "").strip(),
            lang=raw["lang"],
            origin=raw["origin"],
            required_articles=required,
            evidence=evidence,
            acceptable_articles=acceptable,
            preferred_clause=preferred,
            source_url=(raw.get("source_url") or None),
            status=status,
            reviewed_by_human=bool(raw.get("reviewed_by_human", False)),
            notes=str(raw.get("notes", "")),
            tags=tuple(raw.get("tags") or ()),
        )

    def to_dict(self) -> dict:
        """Порядок ключей фиксирован и одинаков для синтетических и реальных
        вопросов: так диффы в git читаются глазами."""
        out: dict = {
            "id": self.id,
            "question": self.question,
            "origin": self.origin,
            "source_url": self.source_url,
            "required_articles": list(self.required_articles),
            "acceptable_articles": list(self.acceptable_articles),
            "preferred_clause": (
                None
                if self.preferred_clause is None
                else {
                    "article": self.preferred_clause.article,
                    "clause": self.preferred_clause.clause,
                }
            ),
            "tags": list(self.tags),
            "lang": self.lang,
            "evidence": [{"article": e.article, "quote": e.quote} for e in self.evidence],
            "status": self.status,
            "reviewed_by_human": self.reviewed_by_human,
        }
        if self.notes:
            out["notes"] = self.notes
        return out


@dataclass(frozen=True)
class EvalDataset:
    """Датасет целиком плюс срезы, в которых он прогоняется."""

    questions: tuple[EvalQuestion, ...]
    path: Path | None = None
    schema_version: str = DATASET_SCHEMA_VERSION

    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def slice(self, lang: Lang) -> EvalDataset:  # noqa: D401
        """Языковой срез.

        Русский и казахский срезы считаются раздельно и в EVALUATION.md идут
        разными строками: смешивать их — значит объяснять изменение метрики
        кроссязычностью там, где менялся retrieval, и наоборот.
        """
        return EvalDataset(
            questions=tuple(q for q in self.ready if q.lang == lang),
            path=self.path,
            schema_version=self.schema_version,
        )

    @property
    def ready(self) -> tuple[EvalQuestion, ...]:
        """Только заполненные вопросы. Всё, что считается, считается по ним."""
        return tuple(q for q in self.questions if not q.is_draft)

    @property
    def stats(self) -> dict[str, int]:
        ready = self.ready
        by_lang = Counter(q.lang for q in ready)
        by_origin = Counter(q.origin for q in ready)
        return {
            "total": len(ready),
            "ru": by_lang["ru"],
            "kk": by_lang["kk"],
            "real": by_origin["real"],
            "synthetic": by_origin["synthetic"],
            "with_preferred_clause": sum(1 for q in ready if q.preferred_clause),
            "reviewed_by_human": sum(1 for q in ready if q.reviewed_by_human),
            "draft_slots": len(self.questions) - len(ready),
        }


@dataclass(frozen=True)
class CompletenessRule:
    """Условия, при которых датасет считается укомплектованным.

    Пока правило не выполнено, baseline не запускается и в EVALUATION.md не
    пишется ни одной цифры. Правило вынесено в конфиг, а не зашито в код,
    но проверяется всегда.
    """

    min_ru: int = 60
    min_kk: int = 15
    min_real: int = 15
    require_human_review: bool = True

    def violations(self, dataset: EvalDataset) -> list[str]:
        s = dataset.stats
        problems: list[str] = []
        if s["ru"] < self.min_ru:
            problems.append(f"русских вопросов {s['ru']}, нужно минимум {self.min_ru}")
        if s["kk"] < self.min_kk:
            problems.append(f"казахских вопросов {s['kk']}, нужно минимум {self.min_kk}")
        if s["real"] < self.min_real:
            problems.append(f"вопросов с origin='real' {s['real']}, нужно минимум {self.min_real}")
        if self.require_human_review:
            unreviewed = [q.id for q in dataset.ready if not q.reviewed_by_human]
            if unreviewed:
                shown = ", ".join(unreviewed[:10])
                tail = f" и ещё {len(unreviewed) - 10}" if len(unreviewed) > 10 else ""
                problems.append(f"не отревьюировано человеком: {shown}{tail}")
        return problems


def load_dataset(path: str | Path) -> EvalDataset:
    """Прочитать JSONL и проверить схему. Дубли id — ошибка."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"датасет не найден: {path}")

    questions: list[EvalQuestion] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: невалидный JSON: {exc}") from exc
        questions.append(EvalQuestion.from_dict(raw, source=f"{path}:{lineno}"))

    duplicates = [qid for qid, n in Counter(q.id for q in questions).items() if n > 1]
    if duplicates:
        raise DatasetError(f"{path}: повторяющиеся id: {sorted(duplicates)}")

    return EvalDataset(questions=tuple(questions), path=path)


def save_dataset(dataset: EvalDataset, path: str | Path) -> None:
    """Записать JSONL, отсортировав по id ради стабильных диффов в git."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(q.to_dict(), ensure_ascii=False)
        for q in sorted(dataset.questions, key=lambda q: q.id)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class CorpusValidationReport:
    """Результат сверки эталона с распарсенным корпусом."""

    missing_articles: list[tuple[str, str]] = field(default_factory=list)
    # Элемент вида «syn_020 (цитата к ст. 85)»: важно, какая именно цитата
    # не нашлась, а не только в каком вопросе.
    quote_not_found: list[str] = field(default_factory=list)
    missing_clauses: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing_articles or self.quote_not_found or self.missing_clauses)


def validate_against_corpus(
    dataset: EvalDataset,
    article_texts: dict[str, str],
    article_clauses: dict[str, tuple[str, ...]] | None = None,
) -> CorpusValidationReport:
    """Проверить, что разметка вообще соответствует тексту кодекса.

    Ловит три класса ошибок разметки: ссылку на несуществующую статью,
    выдуманную цитату и пункт, которого в статье нет. Сравнение цитаты идёт по
    нормализованным пробелам — переносы строк в HTML не должны считаться
    расхождением.

    ``article_texts`` приходит из парсера корпуса; здесь намеренно нет импорта
    парсера, чтобы харнесс оставался от него независимым.
    """

    def squash(text: str) -> str:
        return " ".join(text.split()).lower()

    normalized = {num: squash(text) for num, text in article_texts.items()}
    report = CorpusValidationReport()

    for q in dataset.ready:
        for article in q.required_articles:
            if article not in normalized:
                report.missing_articles.append((q.id, article))

        # Каждая цитата проверяется против СВОЕЙ статьи, а не против всех
        # обязательных сразу: иначе цитата из ст. 99 «доказывала» бы и разметку
        # ст. 100, и неполнота обоснования снова стала бы невидимой.
        for quote in q.evidence:
            haystack = normalized.get(quote.article)
            needle = squash(quote.quote)
            if haystack is not None and needle and needle not in haystack:
                report.quote_not_found.append(f"{q.id} (цитата к ст. {quote.article})")

        # Наличие пункта проверяется по разобранному списку номеров, а не
        # поиском «4.» по тексту: парсер выносит номер пункта в отдельное поле,
        # и в тексте статьи его уже нет. Без списка проверка пропускается —
        # это честнее, чем угадывать по подстроке и врать в обе стороны.
        if article_clauses is not None and q.preferred_clause:
            known = article_clauses.get(q.preferred_clause.article)
            if known is not None and normalize_clause(q.preferred_clause.clause) not in known:
                report.missing_clauses.append((q.id, str(q.preferred_clause)))

    return report
