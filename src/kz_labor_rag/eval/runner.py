"""Прогон eval и запись результата.

Харнесс работает поверх ``Retriever``, ``Generator`` и ``Judge`` и ничего не
знает о том, как устроен поиск. Поэтому он собирается и тестируется до того,
как появится хоть какой-то RAG, и не переписывается на каждой итерации.

Результат каждого прогона — самодостаточный JSON: по нему строка в
EVALUATION.md восстанавливается целиком, включая конфиг, версии промптов и
коммит, на котором прогон сделан.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kz_labor_rag.config import Config
from kz_labor_rag.eval import metrics as M
from kz_labor_rag.eval.dataset import CompletenessRule, EvalDataset, EvalQuestion
from kz_labor_rag.eval.generator import Generation, Generator
from kz_labor_rag.eval.judge import Judge, Judgement
from kz_labor_rag.types import Retriever

RESULT_SCHEMA_VERSION = "1.0"


class DatasetNotReadyError(RuntimeError):
    """Датасет не укомплектован, а гейт включён.

    Пока это исключение летит — цифрам в EVALUATION.md взяться неоткуда, и это
    правильно: baseline, посчитанный на неполном датасете, несравним с
    итерациями, посчитанными на полном.
    """


@dataclass
class QuestionRun:
    """Всё, что произошло с одним вопросом за прогон."""

    question: EvalQuestion
    retrieved: list[dict[str, Any]]
    metrics: M.QuestionMetrics
    generation: Generation | None = None
    judgement: Judgement | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.question.id,
            "question": self.question.question,
            "lang": self.question.lang,
            "origin": self.question.origin,
            "required_articles": list(self.question.required_articles),
            "acceptable_articles": list(self.question.acceptable_articles),
            "retrieved": self.retrieved,
            "metrics": {k: v for k, v in asdict(self.metrics).items() if k not in ("question_id",)},
            "retrieval_failure": self.metrics.is_retrieval_failure,
            "latency_ms": self.latency_ms,
        }
        if self.question.preferred_clause:
            out["preferred_clause"] = str(self.question.preferred_clause)
        if self.generation is not None:
            out["generation"] = {
                "answer": self.generation.answer,
                "cited_articles": list(self.generation.cited_articles),
                "error": self.generation.error,
                "input_tokens": self.generation.input_tokens,
                "output_tokens": self.generation.output_tokens,
            }
        if self.judgement is not None:
            out["judgement"] = {
                "score": self.judgement.score,
                "verdict": self.judgement.verdict,
                "reasoning": self.judgement.reasoning,
                "unsupported_claims": list(self.judgement.unsupported_claims),
                "error": self.judgement.error,
            }
        return out


def _mean(values: Sequence[float]) -> float | None:
    """Среднее по непустой выборке.

    ``None`` вместо нуля на пустой выборке — принципиально: «не измеряли» и
    «измерили и получили ноль» это разные вещи, и склеивать их в агрегате
    значит врать в таблице.
    """
    return statistics.fmean(values) if values else None


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvalRunner:
    """Прогоняет датасет через поиск, генерацию и судью и считает метрики."""

    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        generator: Generator | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.judge = judge
        self.k = int(config.get("eval.k"))

    # --- гейт готовности датасета ---------------------------------------

    def check_dataset_ready(self, dataset: EvalDataset) -> None:
        rules = self.config.section("eval")["completeness"]
        if not rules.get("enforce", True):
            return
        rule = CompletenessRule(
            min_ru=int(rules["min_ru"]),
            min_kk=int(rules["min_kk"]),
            min_real=int(rules["min_real"]),
            require_human_review=bool(rules["require_human_review"]),
        )
        if problems := rule.violations(dataset):
            listed = "\n  - ".join(problems)
            raise DatasetNotReadyError(
                "Датасет не укомплектован, прогон остановлен:\n  - "
                + listed
                + "\n\nЦифры на неполном датасете несравнимы с будущими итерациями. "
                "Доукомплектуйте датасет либо временно снимите eval.completeness.enforce "
                "в конфиге — но тогда результат нельзя писать в EVALUATION.md."
            )

    # --- один вопрос ------------------------------------------------------

    def run_question(self, q: EvalQuestion) -> QuestionRun:
        latency: dict[str, float] = {}

        t0 = time.perf_counter()
        chunks = list(self.retriever.search(q.question, self.k))
        latency["retrieval"] = (time.perf_counter() - t0) * 1000

        ranked = M.rank_articles(chunks)

        generation: Generation | None = None
        judgement: Judgement | None = None
        citation_validity: float | None = None
        faithfulness: float | None = None

        if self.generator is not None:
            t0 = time.perf_counter()
            generation = self.generator.generate(q.question, chunks)
            latency["generation"] = (time.perf_counter() - t0) * 1000
            if generation.ok:
                citation_validity = M.citation_validity(generation.cited_articles, chunks)

                if self.judge is not None:
                    t0 = time.perf_counter()
                    judgement = self.judge.judge(q.question, chunks, generation.answer)
                    latency["judge"] = (time.perf_counter() - t0) * 1000
                    if judgement.ok:
                        faithfulness = judgement.score

        clause_precision = clause_hit = None
        if q.preferred_clause is not None:
            clause_precision = M.clause_precision_at_k(q.preferred_clause, chunks, self.k)
            clause_hit = M.clause_hit_at_k(q.preferred_clause, chunks, self.k)

        qm = M.QuestionMetrics(
            question_id=q.id,
            recall_at_k=M.recall_at_k(q.required_articles, ranked, self.k),
            strict_hit_at_k=M.strict_hit_at_k(q.required_articles, ranked, self.k),
            reciprocal_rank=M.reciprocal_rank(q.required_articles, ranked),
            citation_validity=citation_validity,
            faithfulness=faithfulness,
            clause_precision_at_k=clause_precision,
            clause_hit_at_k=clause_hit,
            retrieved_articles=ranked,
            required_articles=list(q.required_articles),
        )

        return QuestionRun(
            question=q,
            retrieved=[
                {
                    "rank": c.rank,
                    "chunk_id": c.chunk_id,
                    "article": c.article,
                    "clauses": list(c.chunk.clauses),
                    "score": c.score,
                    "preview": c.text[:240],
                }
                for c in chunks
            ],
            metrics=qm,
            generation=generation,
            judgement=judgement,
            latency_ms=latency,
        )

    # --- агрегаты ---------------------------------------------------------

    def aggregate(self, runs: Sequence[QuestionRun]) -> dict[str, Any]:
        if not runs:
            return {"n": 0}
        ms = [r.metrics for r in runs]
        k = self.k
        return {
            "n": len(runs),
            f"recall@{k}": _mean([m.recall_at_k for m in ms]),
            f"strict_hit@{k}": _mean([m.strict_hit_at_k for m in ms]),
            "mrr": _mean([m.reciprocal_rank for m in ms]),
            "citation_validity": _mean(
                [m.citation_validity for m in ms if m.citation_validity is not None]
            ),
            "faithfulness": _mean([m.faithfulness for m in ms if m.faithfulness is not None]),
            f"clause_precision@{k}": _mean(
                [m.clause_precision_at_k for m in ms if m.clause_precision_at_k is not None]
            ),
            f"clause_hit@{k}": _mean(
                [m.clause_hit_at_k for m in ms if m.clause_hit_at_k is not None]
            ),
            "n_with_clause": sum(1 for m in ms if m.clause_precision_at_k is not None),
            "n_judged": sum(1 for m in ms if m.faithfulness is not None),
            "retrieval_failures": sorted(m.question_id for m in ms if m.is_retrieval_failure),
        }

    def _provenance(self) -> dict[str, Any]:
        """Чем построен индекс. Пусто, если поиск такого не сообщает."""
        source = getattr(self.retriever, "provenance", None)
        if source is None:
            return {}
        return dict(source() if callable(source) else source)

    # --- полный прогон ----------------------------------------------------

    def run(self, dataset: EvalDataset, *, enforce_gate: bool = True) -> dict[str, Any]:
        if enforce_gate:
            self.check_dataset_ready(dataset)

        started = datetime.now(UTC)
        t0 = time.perf_counter()
        # Черновые слоты в прогон не идут: у них нет ни вопроса, ни эталона.
        runs = [self.run_question(q) for q in dataset.ready]
        wall = time.perf_counter() - t0

        by_lang = {
            lang: self.aggregate([r for r in runs if r.question.lang == lang])
            for lang in self.config.get("eval.languages")
        }
        by_origin = {
            origin: self.aggregate([r for r in runs if r.question.origin == origin])
            for origin in ("real", "synthetic")
        }

        provenance = self._provenance()
        generator = self.generator.descriptor if self.generator else {}
        judge = self.judge.descriptor if self.judge else {}
        dataset_sha = _file_sha256(dataset.path)

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "metrics_version": M.METRICS_VERSION,
            "timestamp": started.isoformat(),
            "version": self.config.version,
            "description": self.config.get_or("description", ""),
            "git": _git_state(),
            "config_fingerprint": self.config.fingerprint,
            "config": self.config.data,
            # Всё, что обязано совпасть, чтобы два прогона можно было поставить
            # в одну таблицу. Подпись чанкинга берётся из index_meta, то есть
            # из индекса, на котором поиск реально работал: подпись, посчитанная
            # по конфигу, не менялась при смене версии чанкера и пропустила бы
            # перенарезку корпуса.
            "comparability": {
                "metrics_version": M.METRICS_VERSION,
                "chunking_signature": provenance.get("chunking_signature"),
                "embeddings_model": provenance.get("embeddings_model"),
                "dataset_sha256": dataset_sha,
                "generator_prompt": generator.get("prompt"),
                "judge_prompt": judge.get("prompt"),
            },
            "index": provenance,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "dataset": {
                "path": str(dataset.path) if dataset.path else None,
                "sha256": dataset_sha,
                "schema_version": dataset.schema_version,
                "stats": dataset.stats,
            },
            "components": {
                "retriever": {"version": self.retriever.version, **provenance},
                "generator": generator or None,
                "judge": judge or None,
            },
            "aggregates": {
                "primary": by_lang.get(self.config.get("eval.primary_language"), {}),
                "by_language": by_lang,
                "by_origin": by_origin,
            },
            "timing": {"wall_seconds": round(wall, 2)},
            "questions": [r.to_dict() for r in runs],
        }


def save_result(result: dict[str, Any], results_dir: str | Path) -> Path:
    """Записать результат прогона. Перезапись существующего файла запрещена.

    Имя файла содержит и время, и метку версии: историю итераций должно быть
    видно в `ls`, не открывая ни одного JSON.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(result["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    safe_version = str(result["version"]).replace("/", "-").replace(" ", "_")
    path = results_dir / f"{stamp}__{safe_version}.json"
    if path.exists():
        raise FileExistsError(f"результат уже существует: {path}")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
