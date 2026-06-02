from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ABTestETL:
    def __init__(self, raw_path: str) -> None:
        self.raw_path = Path(raw_path)
        self.last_mismatches: pd.DataFrame = pd.DataFrame()

    def extract(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_path, parse_dates=["timestamp"])

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        working = df.copy()
        null_counts = working.isna().sum().to_dict()

        if "timestamp" in working.columns:
            working = working.sort_values("timestamp")

        duplicates_removed = int(working.duplicated(subset=["user_id"]).sum())
        working = working.drop_duplicates(subset=["user_id"], keep="first")

        mismatch_mask = ~(
            ((working["group"] == "control") & (working["landing_page"] == "old_page"))
            | ((working["group"] == "treatment") & (working["landing_page"] == "new_page"))
        )
        self.last_mismatches = working.loc[mismatch_mask].copy()
        mismatches_removed = int(mismatch_mask.sum())
        mismatch_pct = (mismatches_removed / len(df) * 100) if len(df) else 0.0
        working = working.loc[~mismatch_mask].copy()

        issues = {
            "rows_raw": int(len(df)),
            "rows_clean": int(len(working)),
            "duplicates_removed": duplicates_removed,
            "mismatches_removed": mismatches_removed,
            "mismatch_pct": float(mismatch_pct),
            "null_counts": null_counts,
        }
        return working.reset_index(drop=True), issues

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        transformed = df.copy()
        transformed["hour"] = transformed["timestamp"].dt.hour
        transformed["day_of_week"] = transformed["timestamp"].dt.day_name().str[:3]
        transformed["date"] = transformed["timestamp"].dt.date

        min_date = transformed["timestamp"].min()
        max_date = transformed["timestamp"].max()
        duration_days = int((max_date.date() - min_date.date()).days) if len(transformed) else 0
        daily_exposure = float(transformed.groupby("date")["user_id"].count().mean()) if len(transformed) else 0.0

        summary = {"experiment_duration_days": duration_days, "daily_exposure": daily_exposure}
        return transformed, summary

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        extracted = self.extract()
        validated, issues = self.validate(extracted)
        transformed, summary = self.transform(validated)
        issues["transform_summary"] = summary
        return transformed, issues


if __name__ == "__main__":
    import argparse
    from dataclasses import dataclass

    @dataclass
    class ETLResult:
        clean_df: pd.DataFrame
        issues: dict[str, Any]
        mismatch_rows: pd.DataFrame

    def run_etl(input_path: str) -> ETLResult:
        etl = ABTestETL(input_path)
        clean_df, issues = etl.run()
        return ETLResult(clean_df=clean_df, issues=issues, mismatch_rows=etl.last_mismatches)

    parser = argparse.ArgumentParser(description="Run AB test ETL pipeline")
    parser.add_argument("--input", required=True, help="Path to raw ab_data.csv")
    args = parser.parse_args()
    result = run_etl(args.input)
    print(result.issues)
