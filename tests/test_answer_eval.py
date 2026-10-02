"""Агрегаты evals/answer_eval.py на ответах с известным исходом."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "answer_eval.py"
spec = importlib.util.spec_from_file_location("answer_eval", SCRIPT)
answer_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer_eval)


def record(
    unanswerable=False, refused=False, withheld=False, hit=False, valid=1, invalid=0, judge=None
):
    return {
        "unanswerable": unanswerable,
        "refused": refused,
        "withheld": withheld,
        "citation_hit": hit,
        "citations": ["ст. 1"] * valid,
        "invalid_citations": ["ст. 2"] * invalid,
        "latency_ms": {"generation": 1000.0},
        **({"judge": judge} if judge else {}),
    }


def test_answerable_and_unanswerable_are_scored_separately():
    agg = answer_eval.aggregate(
        [
            record(hit=True),
            record(refused=True, withheld=True, valid=0, invalid=1),
            record(unanswerable=True, refused=True, valid=0),
            record(unanswerable=True, valid=1),
        ]
    )
    assert agg["answer_rate"] == 0.5
    assert agg["citation_hit"] == 0.5
    assert agg["correct_refusal"] == 0.5
    assert agg["withheld"] == 0.25
    # 2 верные ссылки из 3: проверка до замены ответа отказом.
    assert agg["citation_validity"] == 2 / 3


def test_judge_scores_are_null_without_a_judge():
    agg = answer_eval.aggregate([record()])
    assert agg["correctness"] is None and agg["n_judged"] == 0


def test_judge_scores_are_averaged():
    agg = answer_eval.aggregate(
        [
            record(judge={"correctness": "correct", "groundedness": "grounded"}),
            record(judge={"correctness": "partially_correct", "groundedness": "ungrounded"}),
        ]
    )
    assert agg["correctness"] == 0.75
    assert agg["groundedness"] == 0.5


def test_label_writes_do_not_clobber_another_session(tmp_path, monkeypatch):
    import json

    path = tmp_path / "labels.jsonl"
    rows = [
        {
            "id": i,
            "type": "fact",
            "question": "q",
            "reference": "r",
            "context": "c",
            "answer": "a",
            "label_correctness": None,
            "label_groundedness": None,
        }
        for i in ("a", "b")
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    monkeypatch.setattr(answer_eval, "LABELS", path)
    # Вторая сессия открыта раньше и держит старую копию; первая уже разметила «a».
    stale = answer_eval.read_labels()
    answer_eval.set_label("a", "correct", "grounded")
    stale[1]["label_correctness"], stale[1]["label_groundedness"] = "incorrect", "ungrounded"
    answer_eval.set_label("b", "incorrect", "ungrounded")
    saved = {r["id"]: r for r in answer_eval.read_labels()}
    assert saved["a"]["label_correctness"] == "correct"
    assert saved["b"]["label_correctness"] == "incorrect"
