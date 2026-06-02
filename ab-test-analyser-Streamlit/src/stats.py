from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


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
        n_c, n_t = sample["n_control"], sample["n_treatment"]
        x_c, x_t = rates["control_conversions"], rates["treatment_conversions"]

        if n_c == 0 or n_t == 0:
            return {
                "z_statistic": 0.0, "p_value": 1.0, "alpha": alpha,
                "is_significant": False, "confidence_level": int((1 - alpha) * 100),
                "confidence_interval_control": (0.0, 0.0),
                "confidence_interval_treatment": (0.0, 0.0),
                "verdict": "No statistically significant difference detected.",
                "warning": "One or both groups have zero samples.",
            }

        p_c, p_t = rates["control_rate"], rates["treatment_rate"]
        p_pool = (x_c + x_t) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        # positive z = treatment better than control
        z_stat = (p_t - p_c) / se if se > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        is_significant = p_value < alpha

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_c = (float(max(0.0, p_c - z_crit * np.sqrt(p_c*(1-p_c)/n_c))), float(min(1.0, p_c + z_crit * np.sqrt(p_c*(1-p_c)/n_c))))
        ci_t = (float(max(0.0, p_t - z_crit * np.sqrt(p_t*(1-p_t)/n_t))), float(min(1.0, p_t + z_crit * np.sqrt(p_t*(1-p_t)/n_t))))

        if is_significant and p_t > p_c:
            verdict = "The new page performs significantly better. Recommend deploying."
        elif is_significant and p_t < p_c:
            verdict = "The new page performs significantly worse. Do not deploy."
        else:
            verdict = "No statistically significant difference detected. Consider running the test longer or increasing sample size."

        return {
            "z_statistic": float(z_stat),
            "p_value": p_value,
            "alpha": float(alpha),
            "is_significant": bool(is_significant),
            "confidence_level": int((1 - alpha) * 100),
            "confidence_interval_control": ci_c,
            "confidence_interval_treatment": ci_t,
            "verdict": verdict,
            "warning": (
                "Sample size is very small (<100 per group); interpret p-values cautiously."
                if min(n_c, n_t) < 100 else None
            ),
        }

    def bayesian_test(self, n_simulations: int = 100000) -> dict[str, Any]:
        sample = self.sample_sizes()
        rates = self.conversion_rates()
        n_c, n_t = sample["n_control"], sample["n_treatment"]
        c_conv, t_conv = rates["control_conversions"], rates["treatment_conversions"]

        if n_c == 0 or n_t == 0:
            return {
                "prob_treatment_better": 0.0, "prob_control_better": 100.0,
                "expected_loss_treatment": 0.0, "expected_loss_control": 0.0,
                "credible_interval_95_control": (0.0, 0.0),
                "credible_interval_95_treatment": (0.0, 0.0),
                "posterior_mean_control": 0.0, "posterior_mean_treatment": 0.0,
                "bayesian_verdict": "Evidence favours the control page (Bayesian)",
            }

        rng = np.random.default_rng(42)
        c_samples = rng.beta(1 + c_conv, 1 + n_c - c_conv, size=n_simulations)
        t_samples = rng.beta(1 + t_conv, 1 + n_t - t_conv, size=n_simulations)
        prob_treatment_better = float((t_samples > c_samples).mean() * 100)

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
            "prob_control_better": float(100 - prob_treatment_better),
            "expected_loss_treatment": float(np.maximum(c_samples - t_samples, 0).mean()),
            "expected_loss_control": float(np.maximum(t_samples - c_samples, 0).mean()),
            "credible_interval_95_control": tuple(np.percentile(c_samples, [2.5, 97.5]).astype(float)),
            "credible_interval_95_treatment": tuple(np.percentile(t_samples, [2.5, 97.5]).astype(float)),
            "posterior_mean_control": float(c_samples.mean()),
            "posterior_mean_treatment": float(t_samples.mean()),
            "bayesian_verdict": verdict,
        }

    def effect_size(self, alpha: float = 0.05) -> dict[str, Any]:
        rates = self.conversion_rates()
        h = 2 * np.arcsin(np.sqrt(rates["treatment_rate"])) - 2 * np.arcsin(np.sqrt(rates["control_rate"]))
        abs_h = abs(h)
        label = "negligible" if abs_h < 0.2 else "small" if abs_h < 0.5 else "medium" if abs_h < 0.8 else "large"
        freq = self.frequentist_test(alpha=alpha)
        note = (
            "Even if statistically significant, the effect size is negligible — "
            "consider whether the observed lift justifies deployment cost."
            if abs_h < 0.2 and freq["is_significant"] else None
        )
        return {"cohens_h": float(h), "effect_size_label": label, "practical_significance_note": note}

    def power_analysis(self, alpha: float = 0.05) -> dict[str, Any]:
        sample = self.sample_sizes()
        rates = self.conversion_rates()
        p_c, p_t = rates["control_rate"], rates["treatment_rate"]

        if sample["n_control"] == 0 or sample["n_treatment"] == 0:
            return {
                "observed_power": 0.0, "required_n_per_group": None,
                "is_adequately_powered": False,
                "power_note": "Power could not be computed because at least one group has no observations.",
            }

        if p_c == p_t:
            return {
                "observed_power": 0.05, "required_n_per_group": None,
                "is_adequately_powered": False,
                "power_note": "Observed effect is zero; no finite sample size can detect a zero effect.",
            }

        # Cohen's h as effect size for two proportions
        h = abs(2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c)))
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta  = stats.norm.ppf(0.8)
        n_obs   = min(sample["n_control"], sample["n_treatment"])

        observed_power = float(stats.norm.cdf(h * np.sqrt(n_obs / 2) - z_alpha))
        required = int(np.ceil(2 * ((z_alpha + z_beta) / max(h, 1e-9)) ** 2))

        return {
            "observed_power": observed_power,
            "required_n_per_group": required,
            "is_adequately_powered": bool(observed_power >= 0.8),
            "power_note": f"Observed power is {observed_power:.1%}, indicating whether your current sample was large enough to reliably detect this effect.",
        }

    def segment_analysis(self) -> pd.DataFrame:
        if "timestamp" not in self.df.columns:
            return pd.DataFrame(columns=["segment", "segment_type", "control_rate", "treatment_rate", "lift", "sample_size"])

        working = self.df.copy()
        working["day_of_week"] = working["timestamp"].dt.day_name().str[:3]
        working["hour"] = working["timestamp"].dt.hour
        working["time_of_day"] = pd.cut(
            working["hour"], bins=[-1, 5, 11, 17, 21, 23],
            labels=["Night", "Morning", "Afternoon", "Evening", "Night_late"],
        ).astype(str).replace({"Night_late": "Night"})

        def summarize(by_col: str, segment_type: str) -> pd.DataFrame:
            records = []
            for seg, grp in working.groupby(by_col):
                ctrl = grp.loc[grp["group"] == "control", "converted"]
                trt  = grp.loc[grp["group"] == "treatment", "converted"]
                ctrl_rate = float(ctrl.mean()) if len(ctrl) else 0.0
                trt_rate  = float(trt.mean())  if len(trt)  else 0.0
                records.append({
                    "segment": seg, "segment_type": segment_type,
                    "control_rate": ctrl_rate, "treatment_rate": trt_rate,
                    "lift": trt_rate - ctrl_rate, "sample_size": int(len(grp)),
                })
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

    def run_full_analysis(self, alpha: float = 0.05, include_bayesian: bool = True, n_simulations: int = 100000) -> dict[str, Any]:
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


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from src.etl import ABTestETL

    parser = argparse.ArgumentParser(description="Run AB test statistical analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--simulations", type=int, default=100000)
    args = parser.parse_args()

    etl = ABTestETL(args.input)
    clean_df, _ = etl.run()
    results = ABTestStats(clean_df).run_full_analysis(alpha=args.alpha, n_simulations=args.simulations)

    if isinstance(results.get("segment_analysis"), pd.DataFrame):
        results["segment_analysis"] = results["segment_analysis"].to_dict(orient="records")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Results written to {args.output}")
