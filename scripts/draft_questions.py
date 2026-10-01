"""Черновик тестового набора: evals/questions.jsonl.

    python scripts/draft_questions.py

Запускается один раз. Дальше файл правится только через ``evals/review.py``,
поэтому поверх существующего файла скрипт ничего не пишет.

Что делает:
1. Переносит русскую часть архивного набора (45 синтетических и 15 реальных
   вопросов) с её разметкой. Отметка ревью переносится как ``verified``.
2. Добавляет вопросы по темам, которых в архиве не было, и вопросы на
   несколько пунктов. Цитата не набирается руками: указывается якорь, а
   фрагмент вырезается из текста статьи (``extract_evidence``), и пункт, в
   котором он лежит, определяется по нему же (``clauses_of_quote``).
3. Добавляет вопросы, ответа на которые в Трудовом кодексе нет. Каждый
   проверен поиском по тексту корпуса: нужной нормы там нет, хотя слова из
   вопроса встречаются.

Все новые вопросы получают ``verified: false``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kz_labor_rag.corpus.evidence import clauses_of_quote, extract_evidence  # noqa: E402
from kz_labor_rag.corpus.parser import parse_file  # noqa: E402
from kz_labor_rag.eval.dataset import (  # noqa: E402
    EvalDataset,
    EvalQuestion,
    save_dataset,
    validate_against_corpus,
)

RAW = "data/raw/adilet_K1500000414_rus.html"
ARCHIVE = "evals/datasets/kz_labor_v1.jsonl"
OUT = "evals/questions.jsonl"

# Тип каждого перенесённого вопроса. multi — ровно те, у кого два и больше
# обязательных пунктов; схема проверяет это сама.
ARCHIVE_TYPES = {
    "real_001": "multi", "real_002": "condition", "real_003": "condition",
    "real_004": "multi", "real_005": "condition", "real_006": "condition",
    "real_007": "condition", "real_008": "condition", "real_009": "number",
    "real_010": "condition", "real_011": "fact", "real_012": "number",
    "real_013": "multi", "real_014": "fact", "real_015": "multi",
    "syn_001": "condition", "syn_002": "condition", "syn_003": "number",
    "syn_004": "condition", "syn_005": "multi", "syn_006": "number",
    "syn_007": "number", "syn_008": "multi", "syn_009": "number",
    "syn_010": "condition", "syn_011": "number", "syn_012": "condition",
    "syn_013": "condition", "syn_014": "fact", "syn_015": "condition",
    "syn_016": "multi", "syn_017": "number", "syn_018": "condition",
    "syn_019": "number", "syn_020": "multi", "syn_021": "number",
    "syn_022": "condition", "syn_023": "fact", "syn_024": "fact",
    "syn_025": "number", "syn_026": "number", "syn_027": "fact",
    "syn_028": "fact", "syn_029": "number", "syn_030": "fact",
    "syn_031": "condition", "syn_032": "fact", "syn_033": "condition",
    "syn_034": "condition", "syn_035": "condition", "syn_036": "number",
    "syn_037": "number", "syn_038": "number", "syn_039": "multi",
    "syn_040": "condition", "syn_041": "condition", "syn_042": "fact",
    "syn_043": "condition", "syn_044": "number", "syn_045": "fact",
}  # fmt: skip

# (id, вопрос, тип, теги, [(статья, якорь)], допустимые статьи)
NEW = [
    ("new_001", "Меня сокращают. За сколько должны предупредить и что заплатят при увольнении?",
     "multi", ["увольнение"],
     [("53", "не менее чем за один месяц"),
      ("131", "в размере средней заработной платы за месяц")], ["52"]),
    ("new_002", "Сколько дней подряд может длиться вахта?",
     "number", ["вахта"],
     [("135", "Продолжительность вахты не может превышать пятнадцать календарных дней")], []),
    ("new_003", "Я беременна, срок четыре месяца. Могут ли меня отправить работать вахтой?",
     "condition", ["вахта"],
     [("135", "беременные женщины со сроком беременности двенадцать и более недель")], ["44"]),
    ("new_004", "Отправляют в командировку на неделю. Сохранится ли зарплата и что ещё мне "
     "обязаны оплатить?",
     "multi", ["командировка"],
     [("127", "сохраняются место работы (должность) и заработная плата"),
      ("127", "суточные за календарные дни нахождения в командировке")], []),
    ("new_005", "У меня сыну два года. Я могу отказаться от командировки?",
     "condition", ["командировка"],
     [("127", "имеющие детей в возрасте до трех лет")], []),
    ("new_006", "Меня незаконно уволили. Сколько у меня времени, чтобы подать в суд на "
     "восстановление?",
     "number", ["трудовой-спор"],
     [("160", "по спорам о восстановлении на работе – один месяц")], ["159"]),
    ("new_007", "Суд восстановил меня на работе. Заплатят ли за время вынужденного прогула и "
     "когда меня обязаны вернуть на место?",
     "multi", ["трудовой-спор"],
     [("161", "но не более чем за шесть месяцев"),
      ("161", "подлежит немедленному исполнению")], []),
    ("new_008", "Случайно разбил рабочий ноутбук. Обязан ли я возместить ущерб и в каком размере?",
     "multi", ["материальная-ответственность"],
     [("123", "обязаны возместить прямой действительный ущерб"),
      ("123", "Материальная ответственность в полном размере ущерба")], ["120"]),
    ("new_009", "С какого возраста можно официально устроиться на работу?",
     "number", ["приём-на-работу"],
     [("31", "достигшими шестнадцатилетнего возраста")], []),
    ("new_010", "Мне 14 лет, хочу летом подработать. Можно ли и кто должен подписать договор?",
     "multi", ["приём-на-работу"],
     [("31", "учащимися, достигшими четырнадцатилетнего возраста"),
      ("31", "наряду с несовершеннолетним трудовой договор должен подписываться")], []),
    ("new_011", "Сколько часов отдыха положено между сменами?",
     "number", ["рабочее-время"],
     [("83", "не может быть менее двенадцати часов")], []),
    ("new_012", "Пришёл на работу выпившим, меня отстранили. Заплатят ли мне за этот день?",
     "multi", ["дисциплина"],
     [("48", "находящегося на работе в состоянии алкогольного"),
      ("48", "не сохраняется заработная плата")], []),
    ("new_013", "Работаю удалённо. Кто должен обеспечить меня компьютером и оплачивать связь?",
     "fact", ["дистанционная-работа"],
     [("138", "обеспечивают работника необходимыми для выполнения трудовых обязанностей "
              "оборудованием")], []),
    ("new_014", "Кто оплачивает больничный и с какого дня?",
     "multi", ["больничный"],
     [("133", "обязан за счет своих средств выплачивать работникам социальное пособие"),
      ("133", "с первого дня нетрудоспособности")], []),
    ("new_015", "Работодатель предложил расторгнуть договор по соглашению сторон. Сколько у меня "
     "времени, чтобы ответить?",
     "number", ["увольнение"],
     [("50", "в течение трех рабочих дней")], []),
    ("new_016", "Мне нужно встать на учёт по беременности, срок восемь недель. Сохранят ли "
     "зарплату за время в поликлинике?",
     "fact", ["беременность"],
     [("126-1", "сохраняются место работы (должность) и средняя заработная плата")], []),
]  # fmt: skip

# (id, вопрос, где ответ на самом деле)
UNANSWERABLE = [
    ("unans_001", "Сколько процентов от зарплаты удерживают в виде индивидуального подоходного "
     "налога?",
     "Ставку ИПН устанавливает Налоговый кодекс РК. В ТК РК подоходного налога нет."),
    ("unans_002", "Какой процент от зарплаты работодатель перечисляет в ЕНПФ как обязательные "
     "пенсионные взносы?",
     "Размер ОПВ устанавливает Социальный кодекс РК. ТК РК упоминает взносы (ст. 35, 113), "
     "но размера не задаёт."),
    ("unans_003", "Во сколько лет женщины в Казахстане выходят на пенсию?",
     "Пенсионный возраст устанавливает Социальный кодекс РК; ТК РК лишь ссылается на него "
     "(ст. 52, 53)."),
    ("unans_004", "Какой размер минимальной зарплаты в 2026 году?",
     "Сумму МЗП устанавливает закон о республиканском бюджете; ст. 104 ТК РК описывает "
     "только порядок."),
    ("unans_005", "Сколько ГФСС платит в месяц по уходу за ребёнком до полутора лет?",
     "Социальную выплату по уходу за ребёнком регулирует Социальный кодекс РК; ст. 100 ТК РК "
     "касается только отпуска."),
    ("unans_006", "Какой штраф получит работодатель за задержку зарплаты?",
     "Штрафы устанавливает Кодекс РК об административных правонарушениях; ст. 14 ТК РК "
     "отсылает к законам."),
    ("unans_007", "Как иностранцу получить вид на жительство в Казахстане?",
     "Регулируется законом о правовом положении иностранцев; ст. 32 ТК РК лишь называет "
     "вид на жительство среди документов."),
    ("unans_008", "Как зарегистрировать ИП и какой налоговый режим выбрать?",
     "Предпринимательский и Налоговый кодексы РК."),
    ("unans_009", "Бывший муж не платит алименты. Как их взыскать?",
     "Кодекс РК о браке (супружестве) и семье, ГПК РК."),
    ("unans_010", "Нужно ли платить налог, если продал квартиру, которой владел меньше года?",
     "Налоговый кодекс РК."),
]  # fmt: skip


def migrate(archive: Path) -> list[EvalQuestion]:
    out = []
    for line in archive.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw.get("lang") != "ru":
            continue
        raw["type"] = ARCHIVE_TYPES[raw["id"]]
        raw["verified"] = raw.pop("reviewed_by_human", False)
        raw["preferred_clause"] = None
        out.append(EvalQuestion.from_dict(raw, source=str(archive)))
    return out


def draft_new(by_number: dict) -> list[EvalQuestion]:
    out = []
    for qid, text, qtype, tags, anchors, acceptable in NEW:
        evidence, clauses = [], {}
        for article, anchor in anchors:
            quote = extract_evidence(by_number[article].full_text, anchor)
            if not any(e["quote"] == quote for e in evidence):
                evidence.append({"article": article, "quote": quote})
            pairs = [(c.number, c.text) for c in by_number[article].clauses]
            for number in clauses_of_quote(pairs, quote):
                clauses[(article, number)] = None
        out.append(
            EvalQuestion.from_dict(
                {
                    "id": qid,
                    "question": text,
                    "lang": "ru",
                    "origin": "synthetic",
                    "type": qtype,
                    "required_articles": list(dict.fromkeys(a for a, _ in anchors)),
                    "required_clauses": [{"article": a, "clause": c} for a, c in clauses],
                    "acceptable_articles": acceptable,
                    "evidence": evidence,
                    "tags": tags,
                },
                source=qid,
            )
        )
    return out


def draft_unanswerable() -> list[EvalQuestion]:
    return [
        EvalQuestion.from_dict(
            {
                "id": qid,
                "question": text,
                "lang": "ru",
                "origin": "synthetic",
                "type": "unanswerable",
                "notes": notes,
                "tags": ["вне-кодекса"],
            },
            source=qid,
        )
        for qid, text, notes in UNANSWERABLE
    ]


def main() -> int:
    out = Path(OUT)
    if out.exists():
        print(f"{out} уже существует и правится через evals/review.py — не перезаписываю.")
        return 1

    code = parse_file(RAW)
    questions = migrate(Path(ARCHIVE)) + draft_new(code.by_number) + draft_unanswerable()
    dataset = EvalDataset(questions=tuple(questions))

    report = validate_against_corpus(
        dataset, code.article_texts(), {a.number: a.clause_numbers for a in code}
    )
    if not report.ok:
        print(f"Разметка не сходится с корпусом: {report}", file=sys.stderr)
        return 1

    save_dataset(dataset, out)
    stats = dataset.stats
    print(f"Записано: {out}")
    print(f"  вопросов: {stats['total']}, проверено: {stats['verified']}")
    print(f"  real / synthetic: {stats['real']} / {stats['synthetic']}")
    by_type: dict[str, int] = {}
    for q in dataset.ready:
        by_type[q.type] = by_type.get(q.type, 0) + 1
    print("  по типам: " + ", ".join(f"{t} {n}" for t, n in sorted(by_type.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
