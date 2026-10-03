"""Вспомогательные функции evals/retrieval_eval.py: конфиг ячейки, сводка, таблица."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kz_labor_rag.config import load_config
from kz_labor_rag.eval.experiments import EMBEDDINGS, derive, table_name

SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "retrieval_eval.py"
spec = importlib.util.spec_from_file_location("retrieval_eval", SCRIPT)
retrieval_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_eval)


def test_derive_overrides_without_touching_the_base():
    base = load_config(apply_env=False)
    before = base.get("embeddings.model")
    config = derive(base, EMBEDDINGS["bge-m3"], version="v", table="exp_t")
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
    assert "| fixed512 | e5-base | dense | 143 | — | 0.500 | 0.250 | — |" in table


def test_context_size_sums_the_top_k_chunks():
    result = result_with([{"retrieval": 1.0}])
    result["questions"][0]["retrieved"] = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    summary = retrieval_eval.summarize(result, k=2, token_lengths={"a": 100, "b": 50, "c": 999})
    assert summary["context_tokens"] == 150


def test_table_names_are_valid_sql_identifiers():
    assert table_name("clause+header", "e5-base") == "exp_clause_header_e5_base"


def run_with(recalls, rrs):
    return {
        "questions": [
            {"id": f"q{i}", "metrics": {"recall_at_k": r, "reciprocal_rank": rr}}
            for i, (r, rr) in enumerate(zip(recalls, rrs, strict=True))
        ]
    }


def test_intervals_and_paired_difference_against_the_baseline():
    rows = [
        {"chunking": "fixed512", "embeddings": "e5-base", "method": "dense"},
        {"chunking": "clause", "embeddings": "e5-base", "method": "dense"},
    ]
    runs = {
        "fixed512/e5-base/dense": run_with([0.0] * 40 + [1.0] * 40, [0.5] * 80),
        "clause/e5-base/dense": run_with([1.0] * 70 + [0.0] * 10, [0.5] * 80),
    }
    retrieval_eval.add_intervals(rows, runs, seed=1)
    assert rows[0]["recall_ci"][0] < 0.5 < rows[0]["recall_ci"][1]
    assert rows[0]["vs_baseline"] == "baseline"
    delta = rows[1]["vs_baseline"]
    assert delta["significant"] and delta["low"] > 0


def test_table_shows_intervals_and_marks_significant_deltas():
    row = {
        "chunking": "clause",
        "embeddings": "e5-base",
        "method": "dense",
        **retrieval_eval.summarize(result_with([{"retrieval": 5.0}]), k=5),
        "recall_ci": (0.4, 0.6),
        "mrr_ci": (0.2, 0.3),
        "vs_baseline": {"mean": 0.1, "low": 0.02, "high": 0.18, "significant": True},
    }
    table = retrieval_eval.render([row], 5, {"n": 2, "real": 1, "synthetic": 1})
    assert "0.500 [0.40, 0.60]" in table
    assert "+0.100 [+0.02, +0.18] *" in table


def test_merge_replaces_known_cells_and_appends_new_ones():
    old_rows = [
        {"chunking": "a", "embeddings": "e", "method": "dense", "mark": "old"},
        {"chunking": "a", "embeddings": "e", "method": "hybrid", "mark": "old"},
    ]
    old_runs = {"a/e/dense": {"v": "old"}, "a/e/hybrid": {"v": "old"}}
    new_rows = [
        {"chunking": "a", "embeddings": "e", "method": "hybrid", "mark": "new"},
        {"chunking": "a", "embeddings": "e", "method": "rerank", "mark": "new"},
    ]
    new_runs = {"a/e/hybrid": {"v": "new"}, "a/e/rerank": {"v": "new"}}
    rows, runs = retrieval_eval.merge_runs(old_rows, old_runs, new_rows, new_runs)
    assert [(r["method"], r["mark"]) for r in rows] == [
        ("dense", "old"),
        ("hybrid", "new"),
        ("rerank", "new"),
    ]
    assert runs["a/e/hybrid"] == {"v": "new"} and runs["a/e/dense"] == {"v": "old"}


def test_stored_runs_are_picked_by_split(tmp_path):
    import json

    folder = tmp_path / "retrieval"
    folder.mkdir()
    for name, payload in {
        "20261001T000000Z.json": {"rows": []},  # прежний формат — dev
        "20261002T000000Z_test.json": {"rows": [], "split": "test"},
        "20261003T000000Z.json": {"rows": [], "split": "dev"},
    }.items():
        (folder / name).write_text(json.dumps(payload))
    assert retrieval_eval.latest_stored(tmp_path, "dev").name == "20261003T000000Z.json"
    assert retrieval_eval.latest_stored(tmp_path, "test").name == "20261002T000000Z_test.json"
    assert retrieval_eval.stored_name("20261003T000000Z", "dev") == "20261003T000000Z.json"
    assert retrieval_eval.stored_name("20261003T000000Z", "test") == "20261003T000000Z_test.json"
    assert retrieval_eval.table_name("test") == "retrieval_table_test.md"
    assert retrieval_eval.table_name("dev") == "retrieval_table.md"


def test_refilter_drops_questions_that_are_no_longer_verified():
    def run(items):
        return {
            "questions": [
                {
                    "id": i,
                    "metrics": {
                        "recall_at_k": r,
                        "reciprocal_rank": rr,
                        "strict_hit_at_k": r,
                        "article_recall_at_k": r,
                    },
                }
                for i, r, rr in items
            ]
        }

    rows = [{"chunking": "a", "embeddings": "e", "method": "m", "n": 3, "recall@5": 0.5}]
    runs = {"a/e/m": run([("q1", 1.0, 1.0), ("q2", 0.0, 0.0), ("bad", 0.5, 0.5)])}
    retrieval_eval.refilter(rows, runs, keep={"q1", "q2"}, k=5)
    assert rows[0]["n"] == 2 and rows[0]["recall@5"] == 0.5 and rows[0]["mrr"] == 0.5
    assert [q["id"] for q in runs["a/e/m"]["questions"]] == ["q1", "q2"]


def test_methods_without_rerank_switch_it_off_even_if_the_base_config_enables_it():
    """Конфиг сервиса включает реранкер; ячейка «dense» или «hybrid» не должна его
    наследовать — иначе baseline таблицы тихо превращается в dense+rerank."""
    from kz_labor_rag.eval.experiments import CHUNKINGS, METHODS

    base = load_config(apply_env=False)
    base.data["retrieval"]["reranker"]["enabled"] = True
    for name, overrides in METHODS.items():
        config = derive(base, CHUNKINGS["fixed512"], overrides, version="v", table="t")
        expected = "rerank" in name
        assert config.get("retrieval.reranker.enabled") is expected, name


def test_only_filter_is_exact_for_full_cell_names_and_a_substring_otherwise():
    full = "fixed512/e5-base/dense"
    assert retrieval_eval.matches(full, "fixed512/e5-base/dense")
    assert not retrieval_eval.matches("fixed512/e5-base/dense+rerank", "fixed512/e5-base/dense")
    assert retrieval_eval.matches("clause+header/e5-base/dense+rerank-k40", "k40")
    assert retrieval_eval.matches(full, "k40,fixed512/e5-base/dense")
    assert retrieval_eval.matches(full, None)
