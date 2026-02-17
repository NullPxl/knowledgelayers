import argparse
from pathlib import Path

import pandas as pd


CATEGORIES = ["EQUIVALENT", "OMITS", "ADDS", "CONTRADICTS", "REFUSES", "UNKNOWN"]


def build_category_area_chart(
    runs_parquet: Path,
    output_html: Path,
    article: str | None = None,
    model: str | None = None,
    run_label: str | None = None,
) -> None:
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "plotly is required for visualization. Install with: pip install plotly"
        ) from exc

    if not runs_parquet.exists():
        raise FileNotFoundError(f"Runs parquet not found: {runs_parquet}")

    df = pd.read_parquet(runs_parquet)
    if article:
        df = df[df["article"] == article]
    if model:
        df = df[df["model"] == model]
    if run_label:
        df = df[df["run_label"] == run_label]
    if df.empty:
        raise ValueError("No rows found after applying filters.")

    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df = df.sort_values(["article", "model", "run_time"])

    long_rows = []
    for _, row in df.iterrows():
        for category in CATEGORIES:
            long_rows.append(
                {
                    "run_time": row["run_time"],
                    "article": row["article"],
                    "model": row["model"],
                    "run_label": row.get("run_label"),
                    "category": category,
                    "rate": row.get(f"rate_{category}", 0.0),
                    "n_questions": row["n_questions"],
                    "dataset": row["dataset"],
                }
            )
    long_df = pd.DataFrame(long_rows)

    fig = px.area(
        long_df,
        x="run_time",
        y="rate",
        color="category",
        line_group="category",
        facet_row="model" if long_df["model"].nunique() > 1 else None,
        hover_data=["article", "model", "run_label", "dataset", "n_questions"],
        title="Category Split Over Time (Stacked Area)",
        category_orders={"category": CATEGORIES},
    )
    fig.update_layout(yaxis_title="Share of Questions", xaxis_title="Run Time", hovermode="x unified")
    fig.update_yaxes(range=[0, 1])

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print(f"Wrote interactive chart: {output_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive stacked-area category chart over time.")
    parser.add_argument("--runs-parquet", default="analytics/eval_runs.parquet", help="Path to run-level parquet file")
    parser.add_argument("--output-html", default="analytics/category_stacked_area.html", help="Output HTML path")
    parser.add_argument("--article", default=None, help="Optional article filter")
    parser.add_argument("--model", default=None, help="Optional model filter")
    parser.add_argument("--run-label", default=None, help="Optional run label filter")
    args = parser.parse_args()

    build_category_area_chart(
        runs_parquet=Path(args.runs_parquet),
        output_html=Path(args.output_html),
        article=args.article,
        model=args.model,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(exc)
        raise SystemExit(1)
