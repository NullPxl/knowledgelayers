import argparse
import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


CATEGORIES = ["EQUIVALENT", "OMITS", "ADDS", "CONTRADICTS", "REFUSES", "UNKNOWN"]
WEEK_SUFFIX_PATTERN = re.compile(r"__week(\d+)$", re.IGNORECASE)


def parse_week_index(dataset_name: str) -> int | None:
    stem = Path(dataset_name).stem
    match = WEEK_SUFFIX_PATTERN.search(stem)
    if match:
        return int(match.group(1))
    return None


def parse_article_name(dataset_name: str) -> str:
    stem = Path(dataset_name).stem
    return WEEK_SUFFIX_PATTERN.sub("", stem)


def infer_run_label(eval_path: Path) -> str | None:
    parts = list(eval_path.parts)
    try:
        runs_idx = parts.index("runs")
    except ValueError:
        return None
    if runs_idx + 1 < len(parts):
        return parts[runs_idx + 1]
    return None


def collect_eval_files(logs_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(logs_dir.rglob("*.eval"))
    return sorted(logs_dir.glob("*.eval"))


def load_eval_rows(eval_path: Path, synthetic_base_date: datetime | None):
    with zipfile.ZipFile(eval_path) as archive:
        header = json.loads(archive.read("header.json"))
        summaries = json.loads(archive.read("summaries.json"))

    run = header["eval"]
    dataset_name = run.get("task_args", {}).get("data") or run.get("task_args_passed", {}).get("data") or ""
    created = datetime.fromisoformat(run["created"])
    week_index = parse_week_index(dataset_name)
    article = parse_article_name(dataset_name)
    run_label = infer_run_label(eval_path)

    if synthetic_base_date is not None and week_index is not None:
        run_time = synthetic_base_date + timedelta(days=(week_index - 1) * 7)
    else:
        run_time = created

    sample_rows = []
    for summary in summaries:
        score = summary["scores"]["structured_multi_judge"]
        meta = score.get("metadata", {})
        consensus = meta.get("consensus", {})
        sample_rows.append(
            {
                "eval_file": str(eval_path),
                "run_id": run.get("run_id"),
                "task_id": run.get("task_id"),
                "model": run.get("model"),
                "dataset": dataset_name,
                "article": article,
                "run_label": run_label,
                "created": created.isoformat(),
                "run_time": run_time.isoformat(),
                "week_index": week_index,
                "sample_id": summary.get("id"),
                "question": summary.get("input"),
                "target": summary.get("target"),
                "answer": score.get("answer"),
                "grade": score.get("value"),
                "category": consensus.get("category", "UNKNOWN"),
                "bias_concern": consensus.get("bias_concern", "UNKNOWN"),
                "tie_break_fields": ",".join(meta.get("tie_break_fields", [])),
            }
        )

    return sample_rows


def build_run_rows(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = samples_df.groupby(
        [
            "eval_file",
            "run_id",
            "task_id",
            "model",
            "dataset",
            "article",
            "run_label",
            "created",
            "run_time",
            "week_index",
        ],
        dropna=False,
    )
    for keys, group in grouped:
        (
            eval_file,
            run_id,
            task_id,
            model,
            dataset,
            article,
            run_label,
            created,
            run_time,
            week_index,
        ) = keys
        n = len(group)
        category_counts = group["category"].value_counts().to_dict()
        grade_counts = group["grade"].value_counts().to_dict()

        row = {
            "eval_file": eval_file,
            "run_id": run_id,
            "task_id": task_id,
            "model": model,
            "dataset": dataset,
            "article": article,
            "run_label": run_label,
            "created": created,
            "run_time": run_time,
            "week_index": week_index,
            "n_questions": n,
            "rate_C": grade_counts.get("C", 0) / n if n else 0.0,
            "rate_P": grade_counts.get("P", 0) / n if n else 0.0,
            "rate_I": grade_counts.get("I", 0) / n if n else 0.0,
        }
        for category in CATEGORIES:
            row[f"rate_{category}"] = category_counts.get(category, 0) / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inspect eval logs into Parquet tables.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing .eval files")
    parser.add_argument("--output-dir", default="analytics", help="Directory to write Parquet files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover .eval files under logs-dir",
    )
    parser.add_argument(
        "--synthetic-base-date",
        default=None,
        help="Optional YYYY-MM-DD base for synthetic weekly time when dataset uses __weekNN suffix",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_base_date = None
    if args.synthetic_base_date:
        synthetic_base_date = datetime.fromisoformat(args.synthetic_base_date).replace(tzinfo=timezone.utc)

    eval_files = collect_eval_files(logs_dir, recursive=args.recursive)
    if not eval_files:
        scope = "recursively" if args.recursive else "directly"
        raise FileNotFoundError(f"No .eval files found {scope} under {logs_dir}")

    sample_rows = []
    for eval_file in eval_files:
        try:
            sample_rows.extend(load_eval_rows(eval_file, synthetic_base_date))
        except Exception as exc:
            print(f"Skipping {eval_file}: {exc}")

    if not sample_rows:
        raise RuntimeError("No sample rows were extracted from eval logs.")

    samples_df = pd.DataFrame(sample_rows)
    runs_df = build_run_rows(samples_df)

    samples_path = output_dir / "eval_samples.parquet"
    runs_path = output_dir / "eval_runs.parquet"
    try:
        samples_df.to_parquet(samples_path, index=False)
        runs_df.to_parquet(runs_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires pyarrow or fastparquet. Install with: pip install pyarrow"
        ) from exc

    print(f"Wrote {len(samples_df)} sample rows -> {samples_path}")
    print(f"Wrote {len(runs_df)} run rows -> {runs_path}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(exc)
        raise SystemExit(1)
