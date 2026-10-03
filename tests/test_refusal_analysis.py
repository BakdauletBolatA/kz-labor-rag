"""Офлайн-симуляция отказа по скору реранкера на сохранённом прогоне ответов."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "refusal_analysis.py"
spec = importlib.util.spec_from_file_location("refusal_analysis", SCRIPT)
refusal_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refusal_analysis)


def rec(qid, unanswerable=False, refused=False, correctness="correct"):
    return {
        "id": qid,
        "type": "fact",
        "unanswerable": unanswerable,
        "refused": refused,
        "withheld": False,
        "citation_hit": not refused,
        "citations": [],
        "invalid_citations": [],
        "answer": "x",
        "latency_ms": {"generation": 1.0},
        "judge": {"correctness": correctness, "groundedness": "grounded"},
    }


def test_low_score_turns_an_answer_into_a_refusal():
    records = [
        rec("a", correctness="incorrect"),
        rec("u", unanswerable=True, correctness="incorrect"),
    ]
    out = refusal_analysis.apply_threshold(records, {"a": 0.9, "u": 0.1}, tau=0.5)
    by_id = {r["id"]: r for r in out}
    assert not by_id["a"]["refused"]
    # Вопрос без ответа, на который система «ответила»: теперь честный отказ.
    assert by_id["u"]["refused"]
    assert by_id["u"]["judge"]["correctness"] == "correct"


def test_refusing_an_answerable_question_is_incorrect():
    out = refusal_analysis.apply_threshold([rec("a")], {"a": 0.1}, tau=0.5)
    assert out[0]["refused"] and out[0]["judge"]["correctness"] == "incorrect"


def test_original_records_are_not_modified():
    records = [rec("a")]
    refusal_analysis.apply_threshold(records, {"a": 0.0}, tau=1.0)
    assert not records[0]["refused"]


def test_sweep_reports_the_tradeoff_per_threshold():
    records = [
        rec("a1", correctness="correct"),
        rec("a2", correctness="incorrect"),
        rec("u1", unanswerable=True, correctness="incorrect"),
    ]
    scores = {"a1": 0.9, "a2": 0.2, "u1": 0.1}
    rows = refusal_analysis.sweep(records, scores, taus=[0.0, 0.5], seed=1)
    base, cut = rows
    assert base["tau"] == 0.0 and base["correctness"] == 1 / 3
    # При пороге 0.5 отказ получают a2 (и так неверный) и u1 (теперь верный).
    assert cut["correctness"] == 2 / 3
    assert cut["refused_answerable"] == 0.5
    assert cut["refused_unanswerable"] == 1.0
