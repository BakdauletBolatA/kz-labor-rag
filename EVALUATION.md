# Evaluation log

The evaluation is described in [README.md](README.md#results): how the test set
was built, what each metric means, the result tables, and what went wrong
along the way. Tables there are refreshed by `python eval.py`; raw per-question
results are in `evals/results/`.

Rules the harness enforces:

- metrics are computed on hand-verified questions only, and every result
  records how many (`dataset.evaluated`);
- `kzrag-eval compare` refuses to put two runs side by side when their metric
  version, window, context format, chunking signature, embedding model, dataset
  or prompts differ;
- prompts are hash-locked in `src/kz_labor_rag/eval/prompts/REGISTRY.json`.

Earlier versions of this file (in Russian, before the clause-level metric) are
in the git history.
