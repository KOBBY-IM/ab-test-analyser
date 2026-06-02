from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.etl import ABTestETL

try:
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_confint, proportion_effectsize, proportions_ztest

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def _manual_two_prop_ztest(counts: np.ndarray, nobs: np.ndarray) -> tuple[float, float]:
    p1 = counts[0] / nobs[0]
    p2 = counts[1] / nobs[1]
    pooled = (counts[0] + counts[1]) / (nobs[0] + nobs[1])
    se = np.sqrt(pooled * (1 - pooled) * ((1 / nobs[0]) + (1 / nobs[1])))
    if se == 0:
        return 0.0, 1.0
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return float(-z_stat), float(p_value)


def _manual_ci(count: int, nobs: int, alpha: float) -> tuple[float, float]:
    if nobs == 0:
        return 0.0, 0.0
    p = count / nobs
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt((p * (1 - p)) / nobs)
    return float(max(0.0, p - z * se)), float(min(1.0, p + z * se))


class ABTestStats:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def _group_arrays(self) -> tuple[pd.Series, pd.Series]:
        control = self.df.loc[self.df["group"] == "control", "converted"]
        treatment = self.df.loc[self.df["group"] == "treatment", "converted"]
        return control, treatment

    def sample_sizes(self) -> dict[str, Any]:
        control, treatment = self._group_arrays()
        n_control = int(len(control))
        n_treatment = int(len(treatment))
        ratio = (n_treatment / n_control) if n_control else 0.0
        return {
            "n_control": n_control,
            "n_treatment": n_treatment,
            "total_users": n_control + n_treatment,
            "imbalance_ratio": ratio,
            "imbalance_flag": bool(ratio > 1.1 or ratio < 0.9) if n_control else True,
        }

    def conversion_rates(self) -> dict[str, Any]:
        control, treatment = self._group_arrays()
        control_conversions = int(control.sum())
        treatment_conversions = int(treatment.sum())
        n_control = int(len(control))
        n_treatment = int(len(treatment))

        control_rate = (control_conversions / n_control) if n_control else 0.0
        treatment_rate = (treatment_conversions / n_treatment) if n_treatment else 0.0
        absolute_lift = treatment_rate - control_rate
        relative_lift = (absolute_lift / control_rate * 100) if control_rate else 0.0

        return {
            "control_rate": float(control_rate),
            "treatment_rate": float(treatment_rate),
            "absolute_lift": float(absolute_lift),
            "relative_lift": float(relative_lift),
            "control_conversions": control_conversions,
            "treatment_conversions": treatment_conversions,
        }

    def frequentist_test(self, alpha: float = 0.05) -> dict[str, Any]:
        sample = self.sample_sizes()
        rates = self.conversion_rates()
        counts = np.array([rates["control_conversions"], rates["treatment_conversions"]])
        nobs = np.array([sample["n_control"], sample["n_treatment"]])

        if np.any(nobs == 0):
            return {
                "z_statistic": 0.0,
                "p_value": 1.0,
                "alpha": alpha,
                "is_significant": False,
                "confidence_level": int((1 - alpha) * 100),
                "confidence_interval_control": (0.0, 0.0),
                "confidence_interval_treatment": (0.0, 0.0),
                "verdict": "No statistically significant difference detected. Results are inconclusive — consider running the test longer or increasing sample size.",
                "warning": "One or both groups have zero samples.",
            }

        if HAS_STATSMODELS:
            z_stat, p_value = proportions_ztest(count=counts, nobs=nobs, alternative="two-sided")
            ci_control = proportion_confint(
                count=rates["control_conversions"],
                nobs=sample["n_control"],
                alpha=alpha,
                method="normal",
            )
            ci_treatment = proportion_confint(
                count=rates["treatment_conversions"],
                nobs=sample["n_treatment"],
                alpha=alpha,
                method="normal",
            )
        else:
            z_stat, p_value = _manual_two_prop_ztest(counts=counts, nobs=nobs)
            ci_control = _manual_ci(rates["control_conversions"], sample["n_control"], alpha)
            ci_treatment = _manual_ci(rates["treatment_conversions"], sample["n_treatment"], alpha)
        is_significant = bool(p_value < alpha)

        if is_significant and rates["treatment_rate"] > rates["control_rate"]:
            verdict = "The new page performs significantly better. Recommend deploying."
        elif is_significant and rates["treatment_rate"] < rates["control_rate"]:
            verdict = "The new page performs significantly worse. Do not deploy."
        else:
            verdict = (
                "No statistically significant difference detected. Results are "
                "inconclusive — consider running the test longer or increasing sample size."
            )

        return {
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "alpha": float(alpha),
            "is_significant": is_significant,
            "confidence_level": int((1 - alpha) * 100),
            "confidence_interval_control": (float(ci_control[0]), float(ci_control[1])),
            "confidence_interval_treatment": (float(ci_treatment[0]), float(ci_treatment[1])),
            "verdict": verdict,
            "warning": (
                "Sample size is very small (<100 per group); interpret p-values cautiously."
                if min(sample["n_control"], sample["n_treatment"]) < 100
                else None
            ),
        }

    def bayesian_test(self, n_simulations: int = 100000) -> dict[str, Any]:
        sample = self.sample_sizes()
        rates = self.conversion_rates()

        n_c = sample["n_control"]
        n_t = sample["n_treatment"]
        c_conv = rates["control_conversions"]
        t_conv = rates["treatment_conversions"]

        if n_c == 0 or n_t == 0:
            empty = np.array([])
            return {
                "prob_treatment_better": 0.0,
                "prob_control_better": 100.0,
                "expected_loss_treatment": 0.0,
                "expected_loss_control": 0.0,
                "credible_interval_95_control": (0.0, 0.0),
                "credible_interval_95_treatment": (0.0, 0.0),
                "bayesian_verdict": "Evidence favours the control page (Bayesian)",
                "simulation_samples_control": empty,
                "simulation_samples_treatment": empty,
            }

        rng = np.random.default_rng(42)
        control_samples = rng.beta(1 + c_conv, 1 + n_c - c_conv, size=n_simulations)
        treatment_samples = rng.beta(1 + t_conv, 1 + n_t - t_conv, size=n_simulations)

        prob_treatment_better = float((treatment_samples > control_samples).mean() * 100)
        prob_control_better = float(100 - prob_treatment_better)
        expected_loss_treatment = float(np.maximum(control_samples - treatment_samples, 0).mean())
        expected_loss_control = float(np.maximum(treatment_samples - control_samples, 0).mean())

        ci_control = tuple(np.percentile(control_samples, [2.5, 97.5]).astype(float))
        ci_treatment = tuple(np.percentile(treatment_samples, [2.5, 97.5]).astype(float))

        if prob_treatment_better > 95:
            verdict = "Strong evidence the new page is better (Bayesian)"
        elif prob_treatment_better > 80:
            verdict = "Moderate evidence the new page is better (Bayesian)"
        elif prob_treatment_better > 50:
            verdict = "Slight lean toward the new page — inconclusive (Bayesian)"
        else:
            verdict = "Evidence favours the control page (Bayesian)"

        return {
            "prob_treatment_better": prob_treatment_better,
            "prob_control_better": prob_control_better,
            "expected_loss_treatment": expected_loss_treatment,
            "expected_loss_control": expected_loss_control,
            "credible_interval_95_control": ci_control,
            "credible_interval_95_treatment": ci_treatment,
            "bayesian_verdict": verdict,
            "simulation_samples_control": control_samples,
            "simulation_samples_treatment": treatment_samples,
        }

    def effect_size(self, alpha: float = 0.05) -> dict[str, Any]:
        rates = self.conversion_rates()
        p_control = rates["control_rate"]
        p_treatment = rates["treatment_rate"]
        h = 2 * np.arcsin(np.sqrt(p_treatment)) - 2 * np.arcsin(np.sqrt(p_control))
        abs_h = abs(h)

        if abs_h < 0.2:
            label = "negligible"
        elif abs_h < 0.5:
            label = "small"
        elif abs_h < 0.8:
            label = "medium"
        else:
            label = "large"

        freq = self.frequentist_test(alpha=alpha)
        note = None
        if abs_h < 0.2 and freq["is_significant"]:
            note = (
                "Even if statistically significant, the effect size is negligible — "
                "consider whether the observed lift justifies deployment cost."
            )
        return {
            "cohens_h": float(h),
            "effect_size_label": label,
            "practical_significance_note": note,
        }

    def power_analysis(self, alpha: float = 0.05) -> dict[str, Any]:
        sample = self.sample_sizes()
        rates = self.conversion_rates()
        p_control = rates["control_rate"]
        p_treatment = rates["treatment_rate"]

        if sample["n_control"] == 0 or sample["n_treatment"] == 0:
            return {
                "observed_power": 0.0,
                "required_n_per_group": None,
                "is_adequately_powered": False,
                "power_note": "Power could not be computed because at least one group has no observations.",
            }

        if p_control == p_treatment:
            observed_power = 0.05
            required = None
            note = "Observed effect is zero; no finite sample size can detect a zero effect."
        else:
            effect = 2 * np.arcsin(np.sqrt(p_treatment)) - 2 * np.arcsin(np.sqrt(p_control))
            if HAS_STATSMODELS:
                analysis = NormalIndPower()
                sm_effect = proportion_effectsize(p_treatment, p_control)
                observed_power = float(
                    analysis.power(
                        effect_size=sm_effect,
                        nobs1=min(sample["n_control"], sample["n_treatment"]),
                        alpha=alpha,
                    )
                )
                required = float(analysis.solve_power(effect_size=sm_effect, power=0.8, alpha=alpha, ratio=1.0))
            else:
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(0.8)
                observed_n = min(sample["n_control"], sample["n_treatment"])
                z_effect = abs(effect) * np.sqrt(observed_n / 2)
                observed_power = float(stats.norm.cdf(z_effect - z_alpha))
                required = float(2 * ((z_alpha + z_beta) / max(abs(effect), 1e-9)) ** 2)
            note = (
                f"Observed power is {observed_power:.1%}, indicating whether your current sample "
                "was large enough to reliably detect this effect."
            )
        return {
            "observed_power": float(observed_power),
            "required_n_per_group": int(np.ceil(required)) if required is not None else None,
            "is_adequately_powered": bool(observed_power >= 0.8),
            "power_note": note,
        }

    def segment_analysis(self) -> pd.DataFrame:
        if "timestamp" not in self.df.columns:
            return pd.DataFrame(
                columns=["segment", "segment_type", "control_rate", "treatment_rate", "lift", "sample_size"]
            )

        working = self.df.copy()
        working["day_of_week"] = working["timestamp"].dt.day_name().str[:3]
        working["hour"] = working["timestamp"].dt.hour
        working["time_of_day"] = pd.cut(
            working["hour"],
            bins=[-1, 5, 11, 17, 21, 23],
            labels=["Night", "Morning", "Afternoon", "Evening", "Night_late"],
        ).astype(str)
        working["time_of_day"] = working["time_of_day"].replace({"Night_late": "Night"})

        def summarize(by_col: str, segment_type: str) -> pd.DataFrame:
            records: list[dict[str, Any]] = []
            for seg, grp in working.groupby(by_col):
                control = grp.loc[grp["group"] == "control", "converted"]
                treatment = grp.loc[grp["group"] == "treatment", "converted"]
                control_rate = float(control.mean()) if len(control) else 0.0
                treatment_rate = float(treatment.mean()) if len(treatment) else 0.0
                records.append(
                    {
                        "segment": seg,
                        "segment_type": segment_type,
                        "control_rate": control_rate,
                        "treatment_rate": treatment_rate,
                        "lift": treatment_rate - control_rate,
                        "sample_size": int(len(grp)),
                    }
                )
            return pd.DataFrame(records)

        day = summarize("day_of_week", "day_of_week")
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day["sort_order"] = day["segment"].map({d: i for i, d in enumerate(day_order)}).fillna(99)
        day = day.sort_values("sort_order").drop(columns=["sort_order"])

        time = summarize("time_of_day", "time_of_day")
        time_order = ["Morning", "Afternoon", "Evening", "Night"]
        time["sort_order"] = time["segment"].map({d: i for i, d in enumerate(time_order)}).fillna(99)
        time = time.sort_values("sort_order").drop(columns=["sort_order"])

        return pd.concat([day, time], ignore_index=True)

    def run_full_analysis(
        self,
        alpha: float = 0.05,
        include_bayesian: bool = True,
        n_simulations: int = 100000,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "sample_sizes": self.sample_sizes(),
            "conversion_rates": self.conversion_rates(),
            "frequentist": self.frequentist_test(alpha=alpha),
            "effect_size": self.effect_size(alpha=alpha),
            "power": self.power_analysis(alpha=alpha),
            "segment_analysis": self.segment_analysis(),
        }
        if include_bayesian:
            results["bayesian"] = self.bayesian_test(n_simulations=n_simulations)
        return results


def _serialize_for_json(results: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(results)
    if "segment_analysis" in serialized and isinstance(serialized["segment_analysis"], pd.DataFrame):
        serialized["segment_analysis"] = serialized["segment_analysis"].to_dict(orient="records")
    if "bayesian" in serialized:
        bayes = dict(serialized["bayesian"])
        for key in ("simulation_samples_control", "simulation_samples_treatment"):
            if isinstance(bayes.get(key), np.ndarray):
                bayes[key] = bayes[key].tolist()
        serialized["bayesian"] = bayes
    return serialized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AB test statistical analysis")
    parser.add_argument("--input", required=True, help="Path to raw ab_data.csv")
    parser.add_argument("--output", required=True, help="Output path for JSON results")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--simulations", type=int, default=100000, help="Bayesian simulation count")
    args = parser.parse_args()

    etl = ABTestETL(args.input)
    clean_df, _issues = etl.run()
    stats_engine = ABTestStats(clean_df)
    results = stats_engine.run_full_analysis(alpha=args.alpha, include_bayesian=True, n_simulations=args.simulations)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(_serialize_for_json(results), indent=2))
    print(f"[TRANSFORM] Results written to {output_path}")
