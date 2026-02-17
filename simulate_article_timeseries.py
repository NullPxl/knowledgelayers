import argparse
import asyncio
from pathlib import Path

from inspect_ai import eval as inspect_eval

from info_compare_eval import compare_answers
from wiki_facts import (
    QUESTION_OUTPUT_DIR,
    clean_source_text,
    generate_validated_qa_pairs,
    sanitize_filename,
    scrape_wikipedia_content,
)
import pandas as pd


def save_pairs_to_csv(pairs, output_path: Path, revision_id: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "input": pair["question"],
                "target": clean_source_text(pair["answer"]),
                "metadata": {"revision_id": revision_id},
            }
            for pair in pairs
        ]
    )
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate repeated QA sets from one article and run evals to simulate time-series runs."
    )
    parser.add_argument("--article", required=True, help="Wikipedia article title")
    parser.add_argument("--runs", type=int, default=5, help="Number of simulated runs")
    parser.add_argument("--qa-per-run", type=int, default=25, help="Questions per generated QA file")
    parser.add_argument("--test-model", default="openai/gpt-5-mini", help="Model under test")
    parser.add_argument("--logs-dir", default="logs", help="Inspect log directory")
    args = parser.parse_args()

    print(f"Scraping article: {args.article}")
    content, revision_id = scrape_wikipedia_content(args.article)
    print(f"Revision: {revision_id}, content length: {len(content)}")

    base_name = sanitize_filename(args.article)
    generated_files: list[Path] = []

    for run_index in range(1, args.runs + 1):
        print(f"\n[{run_index}/{args.runs}] Generating {args.qa_per_run} validated QA pairs...")
        pairs = asyncio.run(generate_validated_qa_pairs(content, args.article, args.qa_per_run))
        file_name = f"{base_name}__week{run_index:02d}.csv"
        csv_path = QUESTION_OUTPUT_DIR / file_name
        save_pairs_to_csv(pairs, csv_path, revision_id)
        generated_files.append(csv_path)
        print(f"Saved dataset: {csv_path}")

        print(f"Running eval for {csv_path.name} on model {args.test_model} ...")
        inspect_eval(
            compare_answers(str(csv_path)),
            model=args.test_model,
            log_dir=args.logs_dir,
        )

    print("\nSimulation complete. Generated datasets:")
    for csv_path in generated_files:
        print(f"- {csv_path}")


if __name__ == "__main__":
    main()
