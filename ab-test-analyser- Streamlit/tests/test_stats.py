from __future__ import annotations

import pandas as pd
import pytest

from src.etl import ABTestETL
from src.stats import ABTestStats


def _df(n_control: int, c_rate: float, n_treatment: int, t_rate: float) -> pd.DataFrame:
    c_conv = int(round(n_control * c_rate))
    t_conv = int(round(n_treatment * t_rate))
    rows = []
    for i in range(n_control):
        rows.append(
            {
                "user_id": i + 1,
                "timestamp": pd.Timestamp("2017-01-01") + pd.Timedelta(hours=i % 24),
                "group": "control",
                "landing_page": "old_page",
                "converted": 1 if i < c_conv else 0,
            }
        )
    for i in range(n_treatment):
        rows.append(
            {
                "user_id": n_control + i + 1,
                "timestamp": pd.Timestamp("2017-01-02") + pd.Timedelta(hours=i % 24),
                "group": "treatment",
                "landing_page": "new_page",
                "converted": 1 if i < t_conv else 0,
            }
        )
    return pd.DataFrame(rows)


def test_conversion_rates_computed_correctly() -> None:
    stats = ABTestStats(_df(100, 0.10, 100, 0.12))
    rates = stats.conversion_rates()
    assert rates["control_rate"] == 0.10
    assert rates["treatment_rate"] == 0.12


def test_absolute_lift_calculation() -> None:
    stats = ABTestStats(_df(200, 0.10, 200, 0.13))
    assert stats.conversion_rates()["absolute_lift"] == 0.03


def test_relative_lift_calculation() -> None:
    stats = ABTestStats(_df(100, 0.10, 100, 0.12))
    assert stats.conversion_rates()["relative_lift"] == pytest.approx(20.0)


def test_frequentist_significant_result() -> None:
    stats = ABTestStats(_df(5000, 0.10, 5000, 0.13))
    freq = stats.frequentist_test()
    assert freq["p_value"] < 0.05
    assert freq["is_significant"] is True


def test_frequentist_insignificant_result() -> None:
    stats = ABTestStats(_df(4000, 0.100, 4000, 0.101))
    freq = stats.frequentist_test()
    assert freq["p_value"] > 0.05
    assert freq["is_significant"] is False


def test_p_value_between_0_and_1() -> None:
    p_value = ABTestStats(_df(300, 0.10, 300, 0.12)).frequentist_test()["p_value"]
    assert 0 <= p_value <= 1


def test_bayesian_probability_between_0_and_1() -> None:
    prob = ABTestStats(_df(300, 0.10, 300, 0.12)).bayesian_test(10000)["prob_treatment_better"]
    assert 0 <= prob <= 100


def test_bayesian_probs_sum_to_100() -> None:
    bayes = ABTestStats(_df(500, 0.10, 500, 0.12)).bayesian_test(10000)
    assert round(bayes["prob_treatment_better"] + bayes["prob_control_better"], 6) == 100.0


def test_cohens_h_positive_lift() -> None:
    h = ABTestStats(_df(300, 0.10, 300, 0.12)).effect_size()["cohens_h"]
    assert h > 0


def test_cohens_h_negative_lift() -> None:
    h = ABTestStats(_df(300, 0.12, 300, 0.10)).effect_size()["cohens_h"]
    assert h < 0


def test_cohens_h_zero_lift() -> None:
    h = ABTestStats(_df(300, 0.12, 300, 0.12)).effect_size()["cohens_h"]
    assert abs(h) < 1e-12


def test_effect_size_label_negligible() -> None:
    label = ABTestStats(_df(1000, 0.10, 1000, 0.101)).effect_size()["effect_size_label"]
    assert label == "negligible"


def test_effect_size_label_small() -> None:
    label = ABTestStats(_df(1000, 0.10, 1000, 0.20)).effect_size()["effect_size_label"]
    assert label == "small"


def test_power_analysis_returns_required_n() -> None:
    out = ABTestStats(_df(500, 0.10, 500, 0.13)).power_analysis()
    assert isinstance(out["required_n_per_group"], int)
    assert out["required_n_per_group"] > 0


def test_observed_power_adequate_large_sample() -> None:
    out = ABTestStats(_df(10000, 0.10, 10000, 0.13)).power_analysis()
    assert out["observed_power"] >= 0.8
    assert out["is_adequately_powered"] is True


def test_segment_analysis_returns_dataframe() -> None:
    segment = ABTestStats(_df(300, 0.10, 300, 0.12)).segment_analysis()
    assert isinstance(segment, pd.DataFrame)


def test_segment_analysis_has_required_columns() -> None:
    segment = ABTestStats(_df(300, 0.10, 300, 0.12)).segment_analysis()
    expected = {"segment", "segment_type", "control_rate", "treatment_rate", "lift", "sample_size"}
    assert expected.issubset(set(segment.columns))


def test_mismatch_detection_in_etl() -> None:
    df = pd.DataFrame(
        [
            {
                "user_id": 1,
                "timestamp": pd.Timestamp("2017-01-01"),
                "group": "control",
                "landing_page": "new_page",
                "converted": 0,
            },
            {
                "user_id": 2,
                "timestamp": pd.Timestamp("2017-01-01"),
                "group": "control",
                "landing_page": "old_page",
                "converted": 1,
            },
        ]
    )
    etl = ABTestETL("data/raw/ab_data.csv")
    clean, issues = etl.validate(df)
    assert len(clean) == 1
    assert issues["mismatches_removed"] == 1


def test_duplicate_removal_in_etl() -> None:
    df = pd.DataFrame(
        [
            {
                "user_id": 1,
                "timestamp": pd.Timestamp("2017-01-01 00:00"),
                "group": "control",
                "landing_page": "old_page",
                "converted": 0,
            },
            {
                "user_id": 1,
                "timestamp": pd.Timestamp("2017-01-01 01:00"),
                "group": "control",
                "landing_page": "old_page",
                "converted": 1,
            },
        ]
    )
    etl = ABTestETL("data/raw/ab_data.csv")
    clean, issues = etl.validate(df)
    assert len(clean) == 1
    assert issues["duplicates_removed"] == 1
    assert clean.iloc[0]["converted"] == 0
