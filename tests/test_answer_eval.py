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


class TestCheckpoint:
    """Прогон ответов идёт часами; обрыв не должен терять готовое."""

    def test_records_survive_and_resume(self, tmp_path):
        ckpt = answer_eval.Checkpoint(tmp_path / "partial.jsonl", key={"cell": "a"})
        ckpt.add({"id": "q1", "answer": "x", "_hits": ["не сериализуется"]})
        again = answer_eval.Checkpoint(tmp_path / "partial.jsonl", key={"cell": "a"})
        assert [r["id"] for r in again.records] == ["q1"]
        assert "_hits" not in again.records[0]

    def test_other_settings_start_from_scratch(self, tmp_path):
        answer_eval.Checkpoint(tmp_path / "partial.jsonl", key={"cell": "a"}).add({"id": "q1"})
        other = answer_eval.Checkpoint(tmp_path / "partial.jsonl", key={"cell": "b"})
        assert other.records == []

    def test_done_removes_the_file(self, tmp_path):
        path = tmp_path / "partial.jsonl"
        ckpt = answer_eval.Checkpoint(path, key={"cell": "a"})
        ckpt.add({"id": "q1"})
        ckpt.done()
        assert not path.exists()


def test_failed_generations_are_excluded_and_counted():
    ok = record(hit=True)
    failed = {**record(), "error": "HTTPError: HTTP Error 500"}
    agg = answer_eval.aggregate([ok, failed])
    # Упавшая генерация — не «ответил без попадания», а «не измерено».
    assert agg["answer_rate"] == 1.0
    assert agg["citation_hit"] == 1.0
    assert agg["n_failed"] == 1


def test_agreement_without_judged_pairs_explains_itself(tmp_path, monkeypatch, capsys):
    import json

    path = tmp_path / "labels.jsonl"
    row = {
        "id": "a",
        "question": "q",
        "reference": "r",
        "context": "c",
        "answer": "a",
        "label_correctness": "correct",
        "label_groundedness": "grounded",
        "judge": {"correctness": None, "groundedness": None, "error": "Connection error"},
    }
    path.write_text(json.dumps(row) + "\n", "utf-8")
    monkeypatch.setattr(answer_eval, "LABELS", path)
    assert answer_eval.agreement_table([row], "gpt-5") is None


def test_retry_picks_failed_generations_and_missing_verdicts():
    good = {**record(judge={"correctness": "correct", "groundedness": "grounded"}), "id": "a"}
    failed = {**record(), "id": "b", "error": "HTTP 500"}
    unjudged = {**record(judge={"correctness": None, "error": "Connection error"}), "id": "c"}
    no_judge = {**record(), "id": "d"}
    records = [good, failed, unjudged, no_judge]
    assert answer_eval.needs_retry(records, judge_on=True) == ["b", "c", "d"]
    assert answer_eval.needs_retry(records, judge_on=False) == ["b"]
