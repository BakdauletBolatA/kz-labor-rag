"""Вспомогательные функции evals/retrieval_eval.py: конфиг ячейки, сводка, таблица."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kz_labor_rag.config import load_config

SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "retrieval_eval.py"
spec = importlib.util.spec_from_file_location("retrieval_eval", SCRIPT)
retrieval_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_eval)


def test_derive_overrides_without_touching_the_base():
    base = load_config(apply_env=False)
    before = base.get("embeddings.model")
    config = retrieval_eval.derive(
        base, retrieval_eval.EMBEDDINGS["bge-m3"], version="v", table="exp_t"
    )
    assert config.get("embeddings.model") == "BAAI/bge-m3"
    assert config.get("vector_store.table") == "exp_t"
    assert config.get("generation.enabled") is False
    assert base.get("embeddings.model") == before


def result_with(latencies):
    return {
        "aggregates": {"primary": {"n": 2, "recall@5": 0.5, "mrr": 0.25, "retrieval_failures": []}},
        "questions": [{"latency_ms": lat} for lat in latencies],
        "index": {"chunks": 143, "chunking_signature": "sig"},
    }


def test_summary_reports_percentiles_per_step_and_skips_generation():
    summary = retrieval_eval.summarize(
        result_with(
            [
                {"retrieval": 10.0, "dense": 8.0, "generation": 0.0},
                {"retrieval": 30.0, "dense": 20.0, "generation": 0.0},
            ]
        ),
        k=5,
    )
    assert summary["latency_ms"]["retrieval"]["p50"] == 20.0
    assert set(summary["latency_ms"]) == {"retrieval", "dense"}
    assert summary["recall@5"] == 0.5


def test_table_states_n_and_marks_missing_steps():
    row = {
        "chunking": "fixed512",
        "embeddings": "e5-base",
        "method": "dense",
        **retrieval_eval.summarize(result_with([{"retrieval": 5.0, "dense": 4.0}]), k=5),
    }
    table = retrieval_eval.render([row], 5, {"n": 2, "real": 1, "synthetic": 1})
    assert "n = 2" in table
    assert "| fixed512 | e5-base | dense | 0.500 | 0.250 | — |" in table
