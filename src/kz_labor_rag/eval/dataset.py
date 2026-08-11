"""Схема и загрузка eval-датасета.

Датасет — JSONL, по одному вопросу на строку, в git. Он появляется до baseline
и до какого-либо RAG: разметка, сделанная после того, как увидели выдачу,
подгоняет метрику под систему, а не измеряет её.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from kz_labor_rag.types import ClauseRef, normalize_clause

Origin = Literal["real", "synthetic"]
Lang = Literal["ru", "kk"]

DATASET_SCHEMA_VERSION = "1.0"


class DatasetError(ValueError):
    """Датасет нарушает схему. Всегда фатально: молча чинить эталон нельзя."""


@dataclass(frozen=True)
class EvalQuestion:
    """Один размеченный вопрос.

    ``evidence_quote`` — дословный фрагмент из корпуса, обосновывающий разметку.
    Он существует не для метрик, а для ревью человеком: по цитате видно, из
    какой нормы взят эталон, не открывая кодекс. Валидатор проверяет, что
    цитата действительно встречается в тексте обязательной статьи.
    """

    id: str
    question: str
    lang: Lang
    origin: Origin
    required_articles: tuple[int, ...]
    evidence_quote: str
    acceptable_articles: tuple[int, ...] = ()
    preferred_clause: ClauseRef | None = None
    reviewed_by_human: bool = False
    notes: str = ""
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict, *, source: str = "<dict>") -> "EvalQuestion":
        def fail(msg: str) -> None:
            raise DatasetError(f"{source}: вопрос {raw.get('id', '<без id>')}: {msg}")

        for key in ("id", "question", "lang", "origin", "required_articles", "evidence_quote"):
            if not raw.get(key):
                fail(f"обязательное поле '{key}' пусто или отсутствует")

        if raw["lang"] not in ("ru", "kk"):
            fail(f"lang='{raw['lang']}', допустимы только 'ru' и 'kk'")
        if raw["origin"] not in ("real", "synthetic"):
            fail(f"origin='{raw['origin']}', допустимы только 'real' и 'synthetic'")

        required = tuple(int(a) for a in raw["required_articles"])
        acceptable = tuple(int(a) for a in raw.get("acceptable_articles") or ())
        overlap = set(required) & set(acceptable)
        if overlap:
            fail(
                f"статьи {sorted(overlap)} перечислены и в required, и в acceptable — "
                "эталон должен быть однозначным"
            )

        preferred = None
        if pc := raw.get("preferred_clause"):
            preferred = ClauseRef(article=int(pc["article"]), clause=str(pc["clause"]))
            if preferred.article not in required:
                fail(
                    f"preferred_clause указывает на статью {preferred.article}, "
                    "которой нет в required_articles"
                )

        return cls(
            id=str(raw["id"]),
            question=str(raw["question"]).strip(),
            lang=raw["lang"],
            origin=raw["origin"],
            required_articles=required,
            evidence_quote=str(raw["evidence_quote"]).strip(),
            acceptable_articles=acceptable,
            preferred_clause=preferred,
            reviewed_by_human=bool(raw.get("reviewed_by_human", False)),
            notes=str(raw.get("notes", "")),
            tags=tuple(raw.get("tags") or ()),
        )

    def to_dict(self) -> dict:
        out: dict = {
            "id": self.id,
            "question": self.question,
            "lang": self.lang,
            "origin": self.origin,
            "required_articles": list(self.required_articles),
            "acceptable_articles": list(self.acceptable_articles),
            "evidence_quote": self.evidence_quote,
            "reviewed_by_human": self.reviewed_by_human,
        }
        if self.preferred_clause is not None:
            out["preferred_clause"] = {
                "article": self.preferred_clause.article,
                "clause": self.preferred_clause.clause,
            }
        if self.notes:
            out["notes"] = self.notes
        if self.tags:
            out["tags"] = list(self.tags)
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

    def slice(self, lang: Lang) -> "EvalDataset":
        """Языковой срез.

        Русский и казахский срезы считаются раздельно и в EVALUATION.md идут
        разными строками: смешивать их — значит объяснять изменение метрики
        кроссязычностью там, где менялся retrieval, и наоборот.
        """
        return EvalDataset(
            questions=tuple(q for q in self.questions if q.lang == lang),
            path=self.path,
            schema_version=self.schema_version,
        )

    @property
    def stats(self) -> dict[str, int]:
        by_lang = Counter(q.lang for q in self.questions)
        by_origin = Counter(q.origin for q in self.questions)
        return {
            "total": len(self.questions),
            "ru": by_lang["ru"],
            "kk": by_lang["kk"],
            "real": by_origin["real"],
            "synthetic": by_origin["synthetic"],
            "with_preferred_clause": sum(1 for q in self.questions if q.preferred_clause),
            "reviewed_by_human": sum(1 for q in self.questions if q.reviewed_by_human),
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
            problems.append(
                f"вопросов с origin='real' {s['real']}, нужно минимум {self.min_real}"
            )
        if self.require_human_review:
            unreviewed = [q.id for q in dataset if not q.reviewed_by_human]
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

    missing_articles: list[tuple[str, int]] = field(default_factory=list)
    quote_not_found: list[str] = field(default_factory=list)
    missing_clauses: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing_articles or self.quote_not_found or self.missing_clauses)


def validate_against_corpus(
    dataset: EvalDataset, article_texts: dict[int, str]
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

    for q in dataset:
        for article in q.required_articles:
            if article not in normalized:
                report.missing_articles.append((q.id, article))

        haystacks = [normalized[a] for a in q.required_articles if a in normalized]
        needle = squash(q.evidence_quote)
        if needle and haystacks and not any(needle in h for h in haystacks):
            report.quote_not_found.append(q.id)

        if q.preferred_clause and q.preferred_clause.article in normalized:
            clause = normalize_clause(q.preferred_clause.clause)
            body = normalized[q.preferred_clause.article]
            if f"{clause}." not in body and f"{clause})" not in body:
                report.missing_clauses.append((q.id, str(q.preferred_clause)))

    return report
