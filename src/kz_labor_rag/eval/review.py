"""Ревью тестового набора человеком.

Показывает вопрос и полный текст пунктов, на которые размечен ответ, и даёт
четыре действия: отметить проверенным, исправить пункты, удалить вопрос,
пропустить. Файл перезаписывается после каждого действия, поэтому ревью можно
бросить на середине и продолжить с того же места.

При исправлении пунктов цитата не набирается руками: в ``evidence`` уходит
начало текста самого пункта, дословно.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from kz_labor_rag.corpus.evidence import MAX_QUOTE_LEN
from kz_labor_rag.corpus.parser import LaborCode
from kz_labor_rag.eval.dataset import (
    DatasetError,
    EvalDataset,
    EvalQuestion,
    load_dataset,
    save_dataset,
)
from kz_labor_rag.types import ClauseRef

SINGLE_TYPES = ("fact", "number", "condition")
HELP = "[v] верно  [e] исправить пункты  [d] удалить  [s] пропустить  [q] выйти"


def clause_quote(text: str) -> str:
    """Дословное начало пункта, не длиннее ``MAX_QUOTE_LEN``, по границе слова."""
    if len(text) <= MAX_QUOTE_LEN:
        return text
    cut = text[:MAX_QUOTE_LEN]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def parse_clause_refs(line: str) -> list[ClauseRef]:
    """``"52/1 83/"`` → пункты. Пустой номер — статья без нумерации пунктов."""
    refs = []
    for token in line.replace(",", " ").split():
        article, sep, clause = token.partition("/")
        if not sep or not article:
            raise ValueError(f"'{token}': нужен формат статья/пункт, например 52/1 или 83/")
        refs.append(ClauseRef(article, clause))
    if not refs:
        raise ValueError("не указано ни одного пункта")
    return list(dict.fromkeys(refs))


def with_clauses(
    question: EvalQuestion, refs: list[ClauseRef], code: LaborCode, qtype: str
) -> EvalQuestion:
    """Вопрос с новым эталоном: статьи, пункты и цитаты выводятся из пунктов."""
    by_number = code.by_number
    evidence = []
    for ref in refs:
        article = by_number.get(ref.article)
        if article is None:
            raise ValueError(f"статьи {ref.article} в кодексе нет")
        texts = {c.number: c.text for c in article.clauses}
        if ref.clause not in texts:
            known = ", ".join(n or "(без номера)" for n in texts)
            raise ValueError(f"в статье {ref.article} нет пункта '{ref.clause}'. Есть: {known}")
        evidence.append({"article": ref.article, "quote": clause_quote(texts[ref.clause])})

    required = list(dict.fromkeys(r.article for r in refs))
    raw = question.to_dict()
    raw.update(
        type=qtype,
        required_articles=required,
        required_clauses=[{"article": r.article, "clause": r.clause} for r in refs],
        acceptable_articles=[a for a in question.acceptable_articles if a not in required],
        evidence=evidence,
        preferred_clause=None,
        verified=False,
    )
    return EvalQuestion.from_dict(raw, source="ревью")


def save_atomically(dataset: EvalDataset, path: Path) -> None:
    """Прерванная запись не должна оставить полфайла вместо набора."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        save_dataset(dataset, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ReviewSession:
    def __init__(
        self,
        path: str | Path,
        code: LaborCode,
        ask: Callable[[str], str] = input,
        say: Callable[[str], None] = print,
    ) -> None:
        self.path = Path(path)
        self.code = code
        self.ask = ask
        self.say = say
        self.dataset = load_dataset(self.path)

    # --- состояние --------------------------------------------------------

    def _replace(self, new: EvalQuestion | None, old_id: str) -> None:
        # Файл перечитывается перед каждой записью: меняется только этот
        # вопрос, а не вся копия, загруженная при старте. Иначе вторая открытая
        # сессия затёрла бы отметки первой.
        self.dataset = load_dataset(self.path)
        kept = tuple(q for q in self.dataset.questions if q.id != old_id)
        questions = kept if new is None else kept + (new,)
        self.dataset = EvalDataset(questions=questions, path=self.path)
        save_atomically(self.dataset, self.path)

    def _get(self, qid: str) -> EvalQuestion | None:
        return next((q for q in self.dataset.questions if q.id == qid), None)

    def status(self) -> str:
        ready = self.dataset.ready
        done = sum(1 for q in ready if q.verified)
        by_type: dict[str, list[int]] = {}
        for q in ready:
            counts = by_type.setdefault(q.type, [0, 0])
            counts[0] += q.verified
            counts[1] += 1
        lines = [f"Проверено {done} из {len(ready)}."]
        lines += [f"  {t:13} {d} / {n}" for t, (d, n) in sorted(by_type.items())]
        return "\n".join(lines)

    # --- показ ------------------------------------------------------------

    def show(self, q: EvalQuestion) -> None:
        pending = sum(1 for x in self.dataset.ready if not x.verified)
        mark = "проверен" if q.verified else f"не проверен, всего осталось {pending}"
        self.say(f"\n{'=' * 78}\n{q.id}  [{q.type}, {q.origin}]  {mark}")
        self.say(f"Вопрос: {q.question}")
        if q.source_url:
            self.say(f"Источник: {q.source_url}")
        if q.is_unanswerable:
            self.say("Ответа в Трудовом кодексе нет. Где он на самом деле:")
            self.say(textwrap.fill(q.notes, 78, initial_indent="  ", subsequent_indent="  "))
            return

        by_number = self.code.by_number
        for ref in q.required_clauses:
            article = by_number.get(ref.article)
            clauses = {c.number: c.text for c in article.clauses} if article else {}
            title = article.title if article else "статьи нет в кодексе"
            body = clauses.get(ref.clause, "(пункт не найден)")
            self.say(f"\n  {ref} — {title}")
            self.say(textwrap.fill(body, 78, initial_indent="    ", subsequent_indent="    "))
        if q.acceptable_articles:
            self.say(f"\nДопустимые статьи (не штрафуются): {', '.join(q.acceptable_articles)}")
        if q.notes:
            self.say(f"Заметки: {q.notes}")

    # --- действия ---------------------------------------------------------

    def edit(self, q: EvalQuestion) -> EvalQuestion:
        line = self.ask("Пункты через пробел, статья/пункт (52/1 53/1; без номера — 83/): ")
        refs = parse_clause_refs(line)
        if len(refs) >= 2:
            qtype = "multi"
        elif q.type in SINGLE_TYPES:
            qtype = q.type
        else:
            qtype = ""
            while qtype not in SINGLE_TYPES:
                qtype = self.ask(f"Тип вопроса ({' / '.join(SINGLE_TYPES)}): ").strip()
        new = with_clauses(q, refs, self.code, qtype)
        self._replace(new, q.id)
        self.say("Пункты обновлены. Проверьте ещё раз и отметьте [v].")
        return new

    def review(self, q: EvalQuestion) -> bool:
        """Один вопрос. Возвращает False, если пользователь вышел."""
        while True:
            self.show(q)
            choice = self.ask(f"\n{HELP}\n> ").strip().lower()
            if choice == "v":
                self._replace(replace(q, verified=True), q.id)
                return True
            if choice == "s":
                return True
            if choice == "q":
                return False
            if choice == "d":
                if self.ask(f"Удалить {q.id} насовсем? [y/N] ").strip().lower() == "y":
                    self._replace(None, q.id)
                    self.say(f"{q.id} удалён.")
                    return True
                continue
            if choice == "e":
                try:
                    q = self.edit(q)
                except (ValueError, DatasetError) as exc:
                    self.say(f"Не получилось: {exc}")
                continue
            self.say("Не понял команду.")

    def run(self, ids: list[str] | None = None) -> None:
        if ids:
            queue = []
            for qid in ids:
                if self._get(qid) is None:
                    self.say(f"Вопроса {qid} нет в {self.path}")
                else:
                    queue.append(qid)
        else:
            queue = [q.id for q in self.dataset.ready if not q.verified]

        for qid in queue:
            self.dataset = load_dataset(self.path)
            q = self._get(qid)
            if q is None:
                continue
            if not self.review(q):
                break
        self.say("\n" + self.status())
