import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from inspect_ai import eval as inspect_eval

from export_eval_parquets import build_run_rows, collect_eval_files, load_eval_rows
from info_compare_eval import compare_answers
from viz_category_area import build_category_area_chart
from wiki_facts import (
    clean_source_text,
    generate_validated_qa_pairs,
    sanitize_filename,
    scrape_wikipedia_content,
)


def _utc_run_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    return slug.lower() or "model"


def _load_articles(cli_articles: list[str] | None, articles_file: str | None) -> list[str]:
    articles = list(cli_articles or [])
    if articles_file:
        file_path = Path(articles_file)
        for line in file_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            articles.append(value)

    deduped = []
    seen = set()
    for article in articles:
        if article not in seen:
            deduped.append(article)
            seen.add(article)
    return deduped


def _save_dataset_csv(
    pairs: list[dict[str, str]],
    output_path: Path,
    article: str,
    revision_id: int | None,
    run_label: str,
    collected_at: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        rows.append(
            {
                "input": pair["question"],
                "target": clean_source_text(pair["answer"]),
                "metadata": {
                    "article": article,
                    "revision_id": revision_id,
                    "run_label": run_label,
                    "collected_at": collected_at,
                },
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _export_analytics(logs_root: Path, output_dir: Path) -> tuple[Path, Path] | None:
    eval_files = collect_eval_files(logs_root, recursive=True)
    if not eval_files:
        return None

    sample_rows = []
    for eval_file in eval_files:
        try:
            sample_rows.extend(load_eval_rows(eval_file, synthetic_base_date=None))
        except Exception as exc:
            print(f"Skipping {eval_file}: {exc}")

    if not sample_rows:
        return None

    samples_df = pd.DataFrame(sample_rows)
    runs_df = build_run_rows(samples_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "eval_samples.parquet"
    runs_path = output_dir / "eval_runs.parquet"
    samples_df.to_parquet(samples_path, index=False)
    runs_df.to_parquet(runs_path, index=False)
    return samples_path, runs_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one weekly batch: collect QA datasets, evaluate model(s), and refresh dashboards."
    )
    parser.add_argument("--article", action="append", help="Wikipedia article title (repeatable)")
    parser.add_argument("--articles-file", default=None, help="Path to newline-delimited article titles")
    parser.add_argument("--qa-per-article", type=int, default=50, help="Questions to generate per article")
    parser.add_argument(
        "--test-model",
        action="append",
        help="Model to evaluate (repeatable). Default: openai/gpt-5-mini",
    )
    parser.add_argument("--output-root", default="weekly_runs", help="Root folder for run artifacts")
    parser.add_argument("--run-label", default=None, help="Optional run label (default: UTC timestamp)")
    args = parser.parse_args()

    articles = _load_articles(args.article, args.articles_file)
    if not articles:
        raise ValueError("No articles provided. Use --article and/or --articles-file.")

    models = args.test_model or ["openai/gpt-5-mini"]
    run_label = args.run_label or _utc_run_label()
    collected_at = datetime.now(timezone.utc).isoformat()

    root = Path(args.output_root)
    datasets_dir = root / "datasets" / run_label
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run label: {run_label}")
    print(f"Articles: {len(articles)} | QA per article: {args.qa_per_article}")
    print(f"Models: {', '.join(models)}")

    dataset_paths: list[Path] = []
    dataset_manifest: list[dict[str, str | int | None]] = []

    for article in articles:
        print(f"\nCollecting article: {article}")
        content, revision_id = scrape_wikipedia_content(article)
        pairs = asyncio.run(generate_validated_qa_pairs(content, article, args.qa_per_article))
        dataset_path = datasets_dir / f"{sanitize_filename(article)}.csv"
        _save_dataset_csv(
            pairs=pairs,
            output_path=dataset_path,
            article=article,
            revision_id=revision_id,
            run_label=run_label,
            collected_at=collected_at,
        )
        dataset_paths.append(dataset_path)
        dataset_manifest.append(
            {
                "article": article,
                "dataset": str(dataset_path),
                "revision_id": revision_id,
                "n_pairs": len(pairs),
            }
        )
        print(f"Saved dataset: {dataset_path}")

    for model_name in models:
        model_slug = _model_slug(model_name)
        model_root = root / "models" / model_slug
        model_run_dir = model_root / "runs" / run_label
        model_logs_dir = model_run_dir / "logs"
        model_logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating model: {model_name} ({model_slug})")
        for dataset_path in dataset_paths:
            print(f"- Eval: {dataset_path.name}")
            inspect_eval(
                compare_answers(str(dataset_path.resolve())),
                model=model_name,
                log_dir=str(model_logs_dir),
            )

        model_manifest = {
            "run_label": run_label,
            "model": model_name,
            "model_slug": model_slug,
            "collected_at": collected_at,
            "datasets": dataset_manifest,
        }
        (model_run_dir / "manifest.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")

        analytics = _export_analytics(model_root / "runs", model_root / "analytics")
        if analytics:
            _, runs_parquet = analytics
            build_category_area_chart(
                runs_parquet=runs_parquet,
                output_html=model_root / "analytics" / "category_stacked_area.html",
            )
            print(f"Updated model analytics: {model_root / 'analytics'}")

    global_analytics = _export_analytics(root / "models", root / "analytics")
    if global_analytics:
        _, global_runs = global_analytics
        build_category_area_chart(
            runs_parquet=global_runs,
            output_html=root / "analytics" / "category_stacked_area.html",
        )
        print(f"Updated global analytics: {root / 'analytics'}")

    run_manifest = {
        "run_label": run_label,
        "collected_at": collected_at,
        "articles": articles,
        "models": models,
        "qa_per_article": args.qa_per_article,
        "datasets_dir": str(datasets_dir),
    }
    (root / "datasets" / run_label / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )
    print("\nWeekly batch complete.")


if __name__ == "__main__":
    main()
