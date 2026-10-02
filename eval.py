"""Run every measurement and rebuild the result tables in README.md.

    python eval.py                  # retrieval table, answer quality, judge agreement
    python eval.py --skip-answers   # retrieval only (no generation, minutes instead of hours)
    python eval.py --readme-only    # just refresh README from the latest tables

Steps:
1. evals/retrieval_eval.py  -> evals/results/retrieval_table.md
2. evals/answer_eval.py     -> evals/results/answer_table.md
3. evals/answer_eval.py --agreement -> evals/results/judge_agreement.md
   (only if evals/manual_labels.jsonl has labels and a judge key is set)
4. README.md blocks between <!-- BEGIN x --> / <!-- END x --> markers.

Needs the database and Ollama running: `docker compose up -d db ollama ollama-pull`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kz_labor_rag.config import load_config, load_env_file
from kz_labor_rag.eval.answer_judge import judge_available
from kz_labor_rag.eval.dataset import load_dataset
from kz_labor_rag.eval.report import dataset_summary, replace_block, strip_generated_comment

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "evals" / "results"
TABLES = {
    "retrieval_table": RESULTS / "retrieval_table.md",
    "answer_table": RESULTS / "answer_table.md",
    "judge_agreement": RESULTS / "judge_agreement.md",
    "answer_comparison": RESULTS / "answer_comparison.md",
}


def run(*args: str) -> None:
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def labels_ready() -> bool:
    path = ROOT / "evals" / "manual_labels.jsonl"
    if not path.exists():
        return False
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return any(r.get("label_correctness") and r.get("label_groundedness") for r in rows)


def refresh_readme() -> None:
    config = load_config()
    readme = ROOT / "README.md"
    text = readme.read_text("utf-8")
    text = replace_block(
        text, "dataset_summary", dataset_summary(load_dataset(config.path_of("eval.dataset")))
    )
    for name, path in TABLES.items():
        content = (
            strip_generated_comment(path.read_text("utf-8"))
            if path.exists()
            else f"_Not measured yet: run `python eval.py` ({path.relative_to(ROOT)})._"
        )
        text = replace_block(text, name, content)
    readme.write_text(text, "utf-8")
    print(f"\nREADME.md tables refreshed from {RESULTS.relative_to(ROOT)}/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument("--readme-only", action="store_true")
    args = parser.parse_args()
    load_env_file()

    if not args.readme_only:
        run("evals/retrieval_eval.py")
        if not args.skip_answers:
            run("evals/answer_eval.py")
            provider = load_config().get("answer_judge.provider")
            if labels_ready() and judge_available(provider) is None:
                run("evals/answer_eval.py", "--agreement")
            else:
                print("\nJudge agreement skipped: no manual labels or no judge key.")
    refresh_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
