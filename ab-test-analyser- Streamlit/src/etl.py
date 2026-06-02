from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ETLResult:
    clean_df: pd.DataFrame
    issues: dict[str, Any]
    summary: dict[str, Any]
    mismatch_rows: pd.DataFrame


class ABTestETL:
    def __init__(self, raw_path: str) -> None:
        self.raw_path = Path(raw_path)
        self.last_issues: dict[str, Any] = {}
        self.last_summary: dict[str, Any] = {}
        self.last_mismatches: pd.DataFrame = pd.DataFrame()

    def extract(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_path, parse_dates=["timestamp"])
        print(f"[EXTRACT] {len(df):,} rows loaded from {self.raw_path}")
        return df

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        print(f"[VALIDATE] Total rows loaded: {len(df):,}")
        working_df = df.copy()

        null_counts = working_df.isna().sum().to_dict()
        print(f"[VALIDATE] Null counts per column: {null_counts}")

        if "timestamp" in working_df.columns:
            working_df = working_df.sort_values("timestamp")

        duplicates_removed = int(working_df.duplicated(subset=["user_id"]).sum())
        if duplicates_removed:
            print(f"[VALIDATE] Duplicate user_ids detected: {duplicates_removed:,}")
        working_df = working_df.drop_duplicates(subset=["user_id"], keep="first")
        print(f"[VALIDATE] Duplicate rows removed: {duplicates_removed:,}")

        mismatch_mask = ~(
            ((working_df["group"] == "control") & (working_df["landing_page"] == "old_page"))
            | ((working_df["group"] == "treatment") & (working_df["landing_page"] == "new_page"))
        )
        mismatch_rows = working_df.loc[mismatch_mask].copy()
        mismatches_removed = int(mismatch_mask.sum())
        mismatch_pct = (mismatches_removed / len(df) * 100) if len(df) else 0.0
        print(
            f"[VALIDATE] Mismatches detected/removed: {mismatches_removed:,} "
            f"({mismatch_pct:.2f}% of raw rows)"
        )
        working_df = working_df.loc[~mismatch_mask].copy()

        converted_invalid = sorted(
            value for value in working_df["converted"].dropna().unique() if value not in (0, 1)
        )
        group_invalid = sorted(
            value
            for value in working_df["group"].dropna().unique()
            if value not in ("control", "treatment")
        )
        if converted_invalid:
            print(f"[VALIDATE] Invalid converted values found: {converted_invalid}")
        if group_invalid:
            print(f"[VALIDATE] Invalid group values found: {group_invalid}")

        control_df = working_df.loc[working_df["group"] == "control"]
        treatment_df = working_df.loc[working_df["group"] == "treatment"]
        control_size = int(len(control_df))
        treatment_size = int(len(treatment_df))
        control_rate = float(control_df["converted"].mean()) if control_size else 0.0
        treatment_rate = float(treatment_df["converted"].mean()) if treatment_size else 0.0

        issues = {
            "rows_raw": int(len(df)),
            "rows_clean": int(len(working_df)),
            "duplicates_removed": duplicates_removed,
            "mismatches_removed": mismatches_removed,
            "mismatch_pct": float(mismatch_pct),
            "control_size": control_size,
            "treatment_size": treatment_size,
            "control_conversion_rate": control_rate,
            "treatment_conversion_rate": treatment_rate,
            "null_counts": null_counts,
            "invalid_converted_values": converted_invalid,
            "invalid_group_values": group_invalid,
        }

        self.last_issues = issues
        self.last_mismatches = mismatch_rows
        return working_df.reset_index(drop=True), issues

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        transformed = df.copy()
        transformed["converted_label"] = transformed["converted"].map(
            {1: "Converted", 0: "Not converted"}
        )
        transformed["group_label"] = transformed["group"].map(
            {
                "control": "Control (old page)",
                "treatment": "Treatment (new page)",
            }
        )
        transformed["hour"] = transformed["timestamp"].dt.hour
        transformed["day_of_week"] = transformed["timestamp"].dt.day_name().str[:3]
        transformed["date"] = transformed["timestamp"].dt.date

        min_date = transformed["timestamp"].min()
        max_date = transformed["timestamp"].max()
        duration_days = int((max_date.date() - min_date.date()).days) if len(transformed) else 0
        daily_exposure = (
            float(transformed.groupby("date")["user_id"].count().mean()) if len(transformed) else 0.0
        )

        converted_only = transformed.loc[transformed["converted"] == 1]
        if len(converted_only):
            peak_hour = int(converted_only["hour"].value_counts().idxmax())
        else:
            peak_hour = None

        is_weekend = transformed["timestamp"].dt.weekday >= 5
        weekend_rate = (
            float(transformed.loc[is_weekend, "converted"].mean())
            if int(is_weekend.sum()) > 0
            else 0.0
        )
        weekday_rate = (
            float(transformed.loc[~is_weekend, "converted"].mean())
            if int((~is_weekend).sum()) > 0
            else 0.0
        )

        summary = {
            "experiment_duration_days": duration_days,
            "daily_exposure": daily_exposure,
            "peak_hour": peak_hour,
            "weekend_vs_weekday_conversion": {
                "weekend_rate": weekend_rate,
                "weekday_rate": weekday_rate,
                "difference": weekend_rate - weekday_rate,
            },
        }
        print(f"[TRANSFORM] Experiment duration: {duration_days} days")
        print(f"[TRANSFORM] Average daily exposure: {daily_exposure:,.2f}")
        print(f"[TRANSFORM] Peak conversion hour: {peak_hour}")
        print(
            "[TRANSFORM] Weekend vs weekday conversion: "
            f"{weekend_rate:.4f} vs {weekday_rate:.4f}"
        )

        self.last_summary = summary
        return transformed, summary

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        extracted = self.extract()
        validated, issues = self.validate(extracted)
        transformed, summary = self.transform(validated)
        issues = {**issues, "transform_summary": summary}
        return transformed, issues


def run_etl(input_path: str) -> ETLResult:
    etl = ABTestETL(input_path)
    clean_df, issues = etl.run()
    return ETLResult(
        clean_df=clean_df,
        issues=issues,
        summary=etl.last_summary,
        mismatch_rows=etl.last_mismatches,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AB test ETL pipeline")
    parser.add_argument("--input", required=True, help="Path to raw ab_data.csv")
    args = parser.parse_args()

    result = run_etl(args.input)
    print("[TRANSFORM] ETL complete")
    print(result.issues)
