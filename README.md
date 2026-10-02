# kz-labor-rag

Question answering over the Labour Code of the Republic of Kazakhstan, with
answers that cite the exact article and clause, an honest "the Code does not
answer this", and an evaluation that anyone can rerun.

The interesting part of the project is not the chatbot but the measurement:
the evaluation harness was built before the RAG pipeline, every metric is
computed by a script in this repository, and the numbers below are pasted into
this file by `eval.py`, not typed by hand.

## Problem

People ask labour-law questions in plain language — "they want to fire me while
I'm on sick leave, is that legal?" — while the answer sits in one clause of a
223-article code written in legal Russian. A useful answer has to:

- find the right **clause**, not just the right article (article 52 lists more
  than twenty grounds for dismissal);
- cite it (`ст. 54 п. 1`) so the reader can check;
- say "В Трудовом кодексе ответа на это не нашлось" when the question belongs
  to another law (taxes, pensions, migration) instead of improvising.

## Architecture

```mermaid
flowchart LR
    subgraph Indexing
        A[adilet.zan.kz HTML] --> B[Parser<br/>section → chapter → article → clause]
        B --> C[Chunker<br/>one chunk per clause + article header]
        C --> D[multilingual-e5-base]
        D --> E[(pgvector)]
        C --> F[BM25 index<br/>pymorphy3 lemmas]
    end
    subgraph Answering
        Q[Question] --> G[Dense search]
        Q --> H[BM25 search]
        E --> G
        F --> H
        G --> I[Reciprocal rank fusion]
        H --> I
        I --> J[Top-5 fragments<br/>labelled with their clauses]
        J --> K[qwen2.5:7b-instruct<br/>via Ollama]
        K --> L{Citation check:<br/>does every cited clause<br/>appear in the fragments?}
        L -- yes --> M[Answer + sources]
        L -- no valid citation --> N[Refusal]
    end
    subgraph Evaluation
        S[evals/questions.jsonl] --> R[evals/review.py<br/>human verification]
        R --> T[evals/retrieval_eval.py]
        R --> U[evals/answer_eval.py<br/>+ LLM judge]
    end
```

The service is a thin FastAPI wrapper (`src/kz_labor_rag/api.py`) over the
library; the evaluation scripts import the same library, so the API and the
measurements cannot drift apart. A cross-encoder reranker
(`BAAI/bge-reranker-v2-m3`) is implemented and measured but switched off in the
service: on CPU it costs most of the request time (see the retrieval table).

## How to run

Requirements: Docker with about 8 GB of memory and 15 GB of disk.

```bash
cp .env.example .env
```

```bash
docker compose up
```

The first start takes a while: the API container downloads the Code from
adilet.zan.kz and verifies its checksum, Ollama pulls `qwen2.5:7b-instruct`
(4.7 GB), and the index is built on CPU. Everything is cached in volumes, so
later starts take seconds. Progress is in `docker compose logs -f api`.

```bash
curl http://localhost:8000/health
```

```bash
curl -N 'http://localhost:8000/ask/stream?q=Меня+хотят+уволить+на+больничном,+так+можно?'
```

| Endpoint | What it returns |
|---|---|
| `GET /search?q=` | top-5 fragments with raw scores and clause labels |
| `GET /ask?q=` | answer, verified citations, `refused` / `withheld` flags, fragments |
| `GET /ask/stream?q=` | the same as server-sent events: `hits`, `token`…, `done` |
| `GET /context?q=` | the exact context the model sees |

In `/ask/stream` citations can only be checked once the answer is complete, so
the authoritative text is `done.answer`: if the model cited nothing it was
shown, `withheld` is `true` and the answer is replaced by the refusal.

**Faster generation on a Mac.** Docker on macOS has no GPU access, so Ollama in
a container runs on CPU. A native Ollama (`brew install ollama`) uses Metal;
point the service at it with `KZRAG_OLLAMA_URL=http://host.docker.internal:11434`.

## Results

All numbers below are produced by `python eval.py`. The evaluation set is still
being verified by hand, and metrics are computed **on verified questions only**,
so the sample sizes are small — treat differences between rows as anecdotes
until the verified count grows. Latency, chunk counts and context sizes do not
depend on the sample size and are meaningful now.

### Test set

<!-- BEGIN dataset_summary -->
86 questions in `evals/questions.jsonl`: 76 answerable, 10 unanswerable; 15 real user questions, 71 written for this set. **84 verified** — metrics are computed on these only: 3 checked by hand, 81 by a model-assisted review pass. Random spot check of the model-reviewed questions by hand: 0 of 0 confirmed.

| type | questions | verified |
|---|---|---|
| fact | 12 | 12 |
| number | 18 | 18 |
| condition | 23 | 22 |
| multi | 23 | 22 |
| unanswerable | 10 | 10 |
<!-- END dataset_summary -->

### Retrieval: chunking × retrieval method

Script: [`evals/retrieval_eval.py`](evals/retrieval_eval.py). Full per-question
output: `evals/results/retrieval/*.json`.

<!-- BEGIN retrieval_table -->
Verified questions with a retrieval gold: n = 3 (real 0, synthetic 3). Latency is measured on CPU after warm-up.

| chunking | embeddings | method | chunks | context tokens@5 | recall@5 | MRR | article_recall@5 | latency p50, ms | latency p95, ms | dense p50 | bm25 p50 | fusion p50 | rerank p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed512 | e5-base | dense | 143 | 2538 | 0.667 | 0.500 | 0.667 | 52.8 | 57.1 | 52.8 | — | — | — |
| fixed512 | e5-base | bm25-lemma | 143 | 2536 | 0.333 | 0.333 | 0.333 | 0.3 | 0.6 | — | 0.3 | — | — |
| fixed512 | e5-base | hybrid | 143 | 2536 | 0.500 | 0.500 | 0.500 | 56.6 | 62.0 | 55.9 | 0.4 | 0.1 | — |
| fixed512 | e5-base | hybrid+rerank | 143 | 2538 | 0.667 | 0.667 | 0.667 | 6895.7 | 6907.2 | 47.5 | 0.4 | 0.1 | 6815.8 |
| fixed512-overlap128 | e5-base | dense | 191 | 2537 | 0.500 | 0.667 | 0.500 | 49.4 | 58.3 | 49.4 | — | — | — |
| fixed512-overlap128 | e5-base | bm25-lemma | 191 | 2538 | 0.333 | 0.167 | 0.333 | 0.3 | 0.4 | — | 0.3 | — | — |
| fixed512-overlap128 | e5-base | hybrid | 191 | 2537 | 0.500 | 0.444 | 0.500 | 50.7 | 53.7 | 50.1 | 0.5 | 0.1 | — |
| fixed512-overlap128 | e5-base | hybrid+rerank | 191 | 2536 | 0.333 | 0.333 | 0.500 | 6585.2 | 6825.5 | 53.1 | 0.5 | 0.1 | 6537.0 |
| clause | e5-base | dense | 767 | 502 | 0.667 | 0.667 | 1.000 | 58.2 | 72.3 | 58.2 | — | — | — |
| clause | e5-base | bm25-lemma | 767 | 496 | 0.333 | 0.333 | 0.333 | 0.4 | 0.7 | — | 0.4 | — | — |
| clause | e5-base | hybrid | 767 | 449 | 0.667 | 0.667 | 0.667 | 58.9 | 64.3 | 57.9 | 0.7 | 0.1 | — |
| clause | e5-base | hybrid+rerank | 767 | 446 | 0.667 | 0.667 | 0.833 | 2072.1 | 2210.9 | 50.5 | 0.7 | 0.1 | 2020.4 |
| article | e5-base | dense | 276 | 1298 | 0.833 | 0.583 | 0.833 | 54.1 | 62.3 | 54.1 | — | — | — |
| article | e5-base | bm25-lemma | 276 | 1609 | 0.333 | 0.333 | 0.333 | 0.3 | 0.5 | — | 0.3 | — | — |
| article | e5-base | hybrid | 276 | 1485 | 0.833 | 0.511 | 0.833 | 50.0 | 51.9 | 49.3 | 0.5 | 0.1 | — |
| article | e5-base | hybrid+rerank | 276 | 1399 | 1.000 | 0.833 | 1.000 | 4608.1 | 4835.5 | 54.7 | 0.5 | 0.1 | 4552.8 |
| clause+header | e5-base | dense | 767 | 537 | 0.667 | 0.667 | 0.667 | 65.8 | 70.9 | 65.8 | — | — | — |
| clause+header | e5-base | bm25-lemma | 767 | 584 | 0.333 | 0.333 | 0.333 | 0.7 | 1.6 | — | 0.7 | — | — |
| clause+header | e5-base | hybrid | 767 | 489 | 0.667 | 0.667 | 0.667 | 53.0 | 59.8 | 51.9 | 0.6 | 0.1 | — |
| clause+header | e5-base | hybrid+rerank | 767 | 714 | 0.667 | 0.667 | 0.667 | 3354.8 | 3393.0 | 58.9 | 0.8 | 0.1 | 3291.7 |
| fixed512 | e5-base | bm25-stem | 143 | 2536 | 0.500 | 0.417 | 0.500 | 0.2 | 0.3 | — | 0.2 | — | — |
| fixed512 | e5-base | dense+rerank | 143 | 2538 | 0.667 | 0.667 | 0.667 | 6854.5 | 6876.5 | 60.7 | — | — | 6797.6 |
| fixed512 | bge-m3 | dense | 143 | 2537 | 0.667 | 0.417 | 0.667 | 140.0 | 147.4 | 140.0 | — | — | — |
<!-- END retrieval_table -->

How to read it:

- `recall@5` and `MRR` count a hit only when a retrieved chunk contains a
  **required clause**; `article_recall@5` is the looser article-level number.
- `context tokens@5` is how much text the top-5 hands to the generator. Longer
  chunks cover more clauses, so recall can be bought with context size —
  compare chunking strategies only together with this column.
- Latency is per query, on CPU, after warm-up.
- `bge-m3` and `e5-base` are compared on byte-identical chunks.

### Answers

Script: [`evals/answer_eval.py`](evals/answer_eval.py).

<!-- BEGIN answer_table -->
Cell: `clause+header/e5-base/hybrid`, generator: `qwen2.5:7b-instruct` (answer_ru@v2). Verified questions: 3 answerable, 0 unanswerable. Judge not run: no API key for `answer_judge.provider` = openai.

| answer_rate | citation_hit | citation_validity | withheld | correct_refusal | correctness | groundedness | n_judged | generation p50, s |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 0.667 | 1.000 | 0.000 | — | — | — | 0 | 130.261 |
<!-- END answer_table -->

- `citation_hit`: at least one cited clause is a required one.
- `citation_validity`: share of the model's citations that point to a clause
  it was actually shown (measured before the citation check).
- `withheld`: answers replaced by a refusal because no citation was valid.
- `correct_refusal`: unanswerable questions the system declined.
- `correctness` (against the required clauses) and `groundedness` (against the
  shown fragments) come from an LLM judge of a different vendor than the
  generator; they are `—` until a judge key is configured.

### Can the judge be trusted?

Agreement between the judge and hand labels on answers from
`evals/manual_labels.jsonl` (labelled with `python evals/answer_eval.py --label`;
the number of labelled answers is in the table):

<!-- BEGIN judge_agreement -->
_Not measured yet: run `python eval.py` (evals/results/judge_agreement.md)._
<!-- END judge_agreement -->

## How the test set was built

- **60 Russian questions carried over** from the first version of the set: 45
  written for it, phrased the way people actually ask, and 15 real questions
  taken from uchet.kz and dogovor24.kz, each with a link to its thread.
- **16 added** for topics the first version did not cover (redundancy,
  rotational work, business trips, labour disputes, liability, remote work,
  sick pay, minors) — 7 of them need two clauses to answer.
- **10 unanswerable questions** whose answers live in other laws (income tax,
  pension contributions, retirement age, the minimum wage amount, migration).
  Several were chosen because the Code mentions the topic without answering it
  (article 104 describes how the minimum wage is set but not its amount), which
  is exactly where a retriever finds plausible but useless text.
- Gold quotes are never typed by hand: an anchor phrase is located in the
  parsed Code and the surrounding sentence is cut out verbatim; the required
  clauses are derived from where those quotes sit. A quote that is not in the
  Code fails the build.
- Every question starts as `verified: false`; only verified questions count.
  `python evals/review.py` shows the question with the full text of its
  clauses and lets the reviewer verify, re-point the clauses, or delete it.
- **Who verified what is recorded per question** (`verified_by`). A few
  questions were checked by hand; the rest went through a model-assisted
  review pass that read every question against the full text of its clauses
  and the neighbouring ones, re-pointed the gold where a key clause was missing
  (the edits and their reasons are in `evals/review_notes.md` and in each
  question's `notes`), and left two disputable questions unverified. To keep
  that pass honest, a seeded random sample of the model-reviewed questions is
  checked by hand (`python evals/review.py --spot-check 15`); the confirmation
  rate is reported in the summary above, and a rejected question drops out of
  the metrics.
- A 15-question Kazakh slice exists in `evals/datasets/kz_labor_v1.jsonl` but
  is frozen: the corpus is Russian, and cross-lingual retrieval is a separate
  project.

## What failed / what I learned

**The metric overstated the baseline.** Recall was counted per article: a
chunk containing clause 2 of article 54 was a hit for a question answered by
clause 1. With 512-token chunks spanning about four articles each, one
retrieved chunk "found" four articles at once. The fix was a failing test
first (right article, wrong clause must score 0), then clause-level recall and
MRR, with article recall kept as a separately named metric and the metrics
version bumped so old and new runs cannot be compared by accident.

**Clause-level recall has its own trap.** A chunk that holds a whole article
covers every clause in it, so the article chunking looks better on recall
partly because it hands the generator more text. The retrieval table therefore
reports context size next to recall.

**Review marks moved to the wrong questions.** The git history showed three
questions reviewed by hand; the dataset file had the marks on two different
ones, changed silently by a later commit. The first baseline ran on the one
question whose review could be confirmed.

**Small data bugs found while building the gold.** A truncated quote spilled
into the next clause and added a wrong clause to the gold; an article header
was labelled like an unnumbered clause and would have counted a header-only
chunk as a hit. Both were caught by a failing test before the fix.

**Latency included model loading.** The first recorded retrieval latency
(`evals/results/*baseline-v0.json`) included loading the embedding weights on
the first query. The runner now warms up before timing.

**Comparing embedding models compared chunkings.** The model's input window
decides where 512-token chunks end, and e5's `passage:` prefix uses tokens that
bge-m3 does not need, so the two models would have been indexed on different
chunks. Chunk boundaries are now set by one model and the script checks that
the texts match.

**A citation is not evidence.** The citation check catches a model citing a
clause it was never shown. It does not catch a confident wrong answer that
cites a fragment it *was* shown but that says something else — in the answer
run, the redundancy question got "two months' notice" with a citation to a
clause about strikes. Only the judge can catch that.

**The rule "no valid citation → refuse" has a cost.** The model sometimes
cites a sub-item (`8)` of clause 1) as if it were clause 8; the check then
withholds an answer that was right. Refusals also came paraphrased ("ответа на
этот вопрос нет") and with a sources line, and were briefly counted as answers.

**CPU is the bottleneck, not retrieval.** The reranker takes most of the
request time on CPU, and answer generation in Docker on CPU takes minutes per
answer (`generation p50` in the answer table). The service runs without the
reranker; on a Mac, native Ollama is the practical option.

**Next steps:** verify the rest of the set and rerun `eval.py`; a prompt that
tells the model to cite clause numbers only from the fragment labels, measured
as its own iteration; reranking only the top 10 to cut its cost.

## Recording a one-minute demo

1. `docker compose up -d` and wait for `docker compose ps` to show `api` as
   healthy (Ollama model pulled, index built).
2. Open `http://localhost:8000/docs` — the endpoint list is the opening shot.
3. Call `/ask/stream` with a real question (the sick-leave dismissal one):
   show fragments arriving, tokens streaming, and the final `done` event with
   `ст. 54 п. 1`.
4. Ask an unanswerable question ("Какая ставка ИПН?") and show the refusal.
5. Show `/context` for the first question: the fragments are labelled with
   their clauses, which is what makes clause citations possible.
6. Finish on this README's retrieval table and the "What failed" section.

Use a native Ollama for the recording so generation takes seconds, not minutes.

## Repository layout

```
config/default.yaml        service config: clause chunks + header, hybrid search
config/baseline.yaml       naive baseline the first runs were recorded with
eval.py                    every measurement + README table refresh
evals/questions.jsonl      the test set
evals/review.py            hand verification of the test set
evals/retrieval_eval.py    chunking × retrieval table
evals/answer_eval.py       answer quality, refusals, judge, judge agreement
evals/results/             tables and raw per-question results
src/kz_labor_rag/
  corpus/                  parser, chunker, verbatim quote extraction
  embeddings/              e5 encoder with a cache keyed by everything that shapes a vector
  retrieval/               pgvector store, dense, BM25, RRF, reranker, factory
  eval/                    metrics, dataset schema, runner, citations, judges, prompts
  api.py                   FastAPI
```

## Development

```bash
make venv
```

```bash
make check
```

Tests against pgvector and against the downloaded Code are skipped when those
are absent; with `docker compose up -d db` and the Code in `data/raw/` the
whole suite runs. Prompts are versioned and hash-locked in
`src/kz_labor_rag/eval/prompts/REGISTRY.json`: editing a prompt without a new
version fails the run.

Reproducing the recorded baseline runs:

```bash
KZRAG_CONFIG=config/baseline.yaml kzrag-eval run
```
