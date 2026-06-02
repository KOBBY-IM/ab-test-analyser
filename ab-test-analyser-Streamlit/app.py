from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta

from src.etl import ABTestETL
from src.stats import ABTestStats

PRIMARY = "#185FA5"
GREY = "#6B6B6B"


def verdict_style(freq: dict) -> tuple[str, str]:
    if freq["is_significant"] and freq["z_statistic"] > 0:
        return "#EAF7EC", "#146C2E"
    if freq["is_significant"] and freq["z_statistic"] < 0:
        return "#FDECEC", "#9F1D1D"
    return "#FFF7E6", "#8A5A00"


def recommendation_text(results: dict) -> tuple[str, list[str]]:
    freq = results["frequentist"]
    bayes = results.get("bayesian", {})
    rates = results["conversion_rates"]
    power = results["power"]
    sample = results["sample_sizes"]
    p_better = bayes.get("prob_treatment_better", 50)

    if freq["is_significant"] and rates["treatment_rate"] > rates["control_rate"] and p_better > 95:
        return (
            f"✅ DEPLOY — Both approaches confirm the new page is significantly better. "
            f"Treatment: {rates['treatment_rate']:.2%} vs control: {rates['control_rate']:.2%} "
            f"(+{rates['absolute_lift']:.2%} absolute, +{rates['relative_lift']:.1f}% relative). "
            f"P(treatment better) = {p_better:.1f}%.",
            [
                "Monitor post-rollout conversion and guardrail metrics daily.",
                "Run a phased rollout to validate sustained impact.",
                "Document experiment setup and expected business value.",
            ],
        )

    if freq["is_significant"] and rates["treatment_rate"] < rates["control_rate"] and p_better < 50:
        return (
            "❌ DO NOT DEPLOY — The new page performs significantly worse. Investigate and redesign before retesting.",
            [
                "Run qualitative UX review and session replay diagnostics.",
                "Analyse drop-off steps by device and traffic source.",
                "Design a new hypothesis and retest with clean instrumentation.",
            ],
        )

    req_n = f"{power['required_n_per_group']:,}" if power["required_n_per_group"] else "unknown"
    return (
        f"⏸️ CONTINUE TESTING — No significant difference detected. "
        f"Current power: {power['observed_power']:.0%}. For 80% power you need {req_n} users per group "
        f"(currently {sample['n_control']:,}). Run longer or redesign the experiment.",
        [
            "Increase sample size and predefine a stopping rule.",
            "Audit event instrumentation and funnel consistency.",
            "Test a stronger variant with larger expected lift.",
        ],
    )


def build_app() -> None:
    st.set_page_config(page_title="A/B Test Analyser", page_icon="🧪", layout="wide")
    st.title("🧪 A/B Test Analyser")

    st.sidebar.header("Data source")
    use_kaggle = st.sidebar.toggle("Use Kaggle dataset", value=True)
    uploaded = None
    if not use_kaggle:
        uploaded = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])
        with st.sidebar.expander("Schema requirements"):
            st.write("Columns required: user_id, timestamp, group, landing_page, converted")
            st.write('Valid groups: "control", "treatment"')
            st.write('Valid pages: "old_page", "new_page"')
            st.write("Converted values: 0 or 1")

    st.sidebar.header("Test parameters")
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.05
    alpha = st.sidebar.slider("Significance level α", 0.01, 0.10, float(st.session_state.alpha), 0.01)
    st.session_state.alpha = alpha

    st.sidebar.header("Analysis options")
    include_bayes = st.sidebar.checkbox("Include Bayesian analysis", value=True)
    include_segments = st.sidebar.checkbox("Include segment analysis", value=True)
    n_sim = st.sidebar.selectbox("Number of Bayesian simulations", [10000, 50000, 100000], index=2)

    if use_kaggle:
        _here = Path(__file__).parent
        data_path = _here / "data/raw/ab_data.csv"
        if not data_path.exists():
            fallback = _here / "ab_data.csv"
            data_path = fallback if fallback.exists() else data_path
        if not data_path.exists():
            st.error("Dataset not found. Place `ab_data.csv` in `data/raw/` or upload a CSV.")
            st.stop()
        etl = ABTestETL(str(data_path))
        raw_df = etl.extract()
    else:
        if uploaded is None:
            st.info("Upload a CSV to start analysis.")
            st.stop()
        raw_df = pd.read_csv(uploaded, parse_dates=["timestamp"])
        etl = ABTestETL(raw_path="uploaded_file.csv")

    clean_df, issues = etl.validate(raw_df)
    clean_df, transform_summary = etl.transform(clean_df)
    issues["transform_summary"] = transform_summary

    stats_engine = ABTestStats(clean_df)
    results = stats_engine.run_full_analysis(alpha=alpha, include_bayesian=include_bayes, n_simulations=n_sim)

    st.header("Section 1: Data Quality Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows loaded", f"{issues['rows_raw']:,}")
    c2.metric("Mismatches removed", f"{issues['mismatches_removed']:,}", f"{issues['mismatch_pct']:.2f}%")
    c3.metric("Duplicates removed", f"{issues['duplicates_removed']:,}")
    c4.metric("Clean rows used", f"{issues['rows_clean']:,}")

    with st.expander("View data quality details"):
        details = pd.DataFrame(
            [
                {"issue_type": "Null values", "count": int(sum(issues["null_counts"].values())), "action_taken": "Retained unless critical"},
                {"issue_type": "Duplicate user IDs", "count": issues["duplicates_removed"], "action_taken": "Kept first by timestamp"},
                {"issue_type": "Group/Page mismatch", "count": issues["mismatches_removed"], "action_taken": "Removed before analysis"},
            ]
        )
        details["pct_total"] = details["count"] / max(issues["rows_raw"], 1) * 100
        st.dataframe(details[["issue_type", "count", "pct_total", "action_taken"]], use_container_width=True)
        st.caption("Sample of removed mismatch rows (first 10)")
        st.dataframe(etl.last_mismatches.head(10), use_container_width=True)

    st.header("Section 2: Experiment Overview")
    left, right = st.columns(2)
    sample = results["sample_sizes"]
    rates = results["conversion_rates"]
    with left:
        st.metric("Control group size (n)", f"{sample['n_control']:,}")
        st.metric("Treatment group size (n)", f"{sample['n_treatment']:,}")
        st.metric("Control conversion rate", f"{rates['control_rate']:.2%}")
        st.metric("Treatment conversion rate", f"{rates['treatment_rate']:.2%}")
        color = "green" if rates["absolute_lift"] >= 0 else "red"
        st.markdown(
            f"**Absolute lift:** :{color}[{rates['absolute_lift']:+.2%}]  \n"
            f"**Relative lift:** :{color}[{rates['relative_lift']:+.2f}%]"
        )

    with right:
        fig = go.Figure()
        fig.add_bar(x=["Control", "Treatment"], y=[rates["control_rate"], rates["treatment_rate"]], marker_color=[GREY, PRIMARY])
        fig.add_hline(y=rates["control_rate"], line_dash="dash", line_color=GREY)
        fig.update_traces(text=[f"{rates['control_rate']:.2%}", f"{rates['treatment_rate']:.2%}"], textposition="outside")
        fig.update_layout(title="Conversion Rate: Control vs Treatment", template="plotly_white", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Section 3: Statistical Results")
    freq_tab, bayes_tab = st.tabs(["Frequentist", "Bayesian"])
    freq = results["frequentist"]
    with freq_tab:
        bg, fg = verdict_style(freq)
        st.markdown(
            f"<div style='background:{bg};padding:16px;border-radius:8px;border-left:6px solid {fg};'>"
            f"<h3 style='color:{fg};margin:0;'>{freq['verdict']}</h3>"
            f"<p style='margin:8px 0 0 0;'>p-value: {freq['p_value']:.4f} | z-statistic: {freq['z_statistic']:.4f} | Significance level: α = {freq['alpha']:.2f}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if freq.get("warning"):
            st.warning(freq["warning"])
        lcol, rcol = st.columns(2)
        with lcol:
            ci_fig = go.Figure()
            ci_fig.add_trace(
                go.Scatter(
                    x=[rates["control_rate"], rates["treatment_rate"]],
                    y=["Control", "Treatment"],
                    mode="markers",
                    marker=dict(color=[GREY, PRIMARY], size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[
                            freq["confidence_interval_control"][1] - rates["control_rate"],
                            freq["confidence_interval_treatment"][1] - rates["treatment_rate"],
                        ],
                        arrayminus=[
                            rates["control_rate"] - freq["confidence_interval_control"][0],
                            rates["treatment_rate"] - freq["confidence_interval_treatment"][0],
                        ],
                    ),
                )
            )
            ci_fig.update_layout(title="95% Confidence Intervals", template="plotly_white", xaxis_tickformat=".1%")
            st.plotly_chart(ci_fig, use_container_width=True)
            overlap = not (
                freq["confidence_interval_control"][1] < freq["confidence_interval_treatment"][0]
                or freq["confidence_interval_treatment"][1] < freq["confidence_interval_control"][0]
            )
            if overlap:
                st.caption("Overlapping CIs — inconclusive")
        with rcol:
            ratio = min(freq["p_value"] / max(alpha, 1e-9), 1.0)
            st.progress(float(1 - ratio))
            st.write(f"**p = {freq['p_value']:.4f}**")
            threshold_note = "below" if freq["p_value"] < alpha else "above"
            st.write(
                "There is a "
                f"{freq['p_value']*100:.1f}% probability of observing this difference "
                "(or larger) if there were truly no difference between the pages. "
                f"This is {threshold_note} our {alpha*100:.0f}% threshold."
            )

        effect = results["effect_size"]
        st.subheader("Effect size")
        st.write(f"Cohen's h: **{effect['cohens_h']:.4f}** ({effect['effect_size_label']})")
        if effect["practical_significance_note"]:
            st.warning(effect["practical_significance_note"])

        power = results["power"]
        st.subheader("Power analysis")
        st.write(f"Observed power: **{power['observed_power']:.1%}**")
        st.progress(float(min(power["observed_power"], 1.0)))
        st.write(f"Required sample size per group (80% power): **{power['required_n_per_group']}**")
        st.write(f"Your test {'is' if power['is_adequately_powered'] else 'is not'} adequately powered.")
        st.caption(power["power_note"])

    with bayes_tab:
        if not include_bayes:
            st.info("Enable Bayesian analysis from the sidebar to view this tab.")
        else:
            bayes = results["bayesian"]
            bayes_bg = "#EAF7EC" if bayes["prob_treatment_better"] > 80 else "#FFF7E6" if bayes["prob_treatment_better"] > 50 else "#FDECEC"
            st.markdown(
                f"<div style='background:{bayes_bg};padding:16px;border-radius:8px;border-left:6px solid {PRIMARY};'>"
                f"<h3 style='margin:0;'>{bayes['bayesian_verdict']}</h3>"
                f"<p style='margin:8px 0 0 0;'>Probability treatment is better: {bayes['prob_treatment_better']:.1f}%</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            lcol, rcol = st.columns(2)
            with lcol:
                x = np.linspace(0.05, 0.20, 1000)
                sample = results["sample_sizes"]
                conv = results["conversion_rates"]
                control_pdf = beta.pdf(x, 1 + conv["control_conversions"], 1 + sample["n_control"] - conv["control_conversions"])
                treatment_pdf = beta.pdf(
                    x, 1 + conv["treatment_conversions"], 1 + sample["n_treatment"] - conv["treatment_conversions"]
                )
                pdf_fig = go.Figure()
                pdf_fig.add_trace(go.Scatter(x=x, y=control_pdf, fill="tozeroy", name="Control", line=dict(color=GREY)))
                pdf_fig.add_trace(go.Scatter(x=x, y=treatment_pdf, fill="tozeroy", name="Treatment", line=dict(color=PRIMARY)))
                pdf_fig.add_vline(x=bayes["posterior_mean_control"], line_dash="dash", line_color=GREY)
                pdf_fig.add_vline(x=bayes["posterior_mean_treatment"], line_dash="dash", line_color=PRIMARY)
                pdf_fig.update_layout(
                    title="Posterior Distributions — Beta-Binomial Model",
                    template="plotly_white",
                    xaxis_title="Conversion rate",
                )
                pdf_fig.update_xaxes(tickformat=".1%")
                st.plotly_chart(pdf_fig, use_container_width=True)
                st.caption(f"P(treatment > control) = {bayes['prob_treatment_better']:.1f}%")
            with rcol:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=bayes["prob_treatment_better"],
                        number={"suffix": "%"},
                        title={"text": "Probability new page is better"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "steps": [
                                {"range": [0, 50], "color": "#FDECEC"},
                                {"range": [50, 80], "color": "#FFF7E6"},
                                {"range": [80, 95], "color": "#EAF7EC"},
                                {"range": [95, 100], "color": "#D1F2D8"},
                            ],
                            "bar": {"color": PRIMARY},
                        },
                    )
                )
                gauge.update_layout(template="plotly_white")
                st.plotly_chart(gauge, use_container_width=True)

            loss_df = pd.DataFrame(
                [
                    {
                        "Decision": "If we deploy treatment",
                        "Expected loss rate": bayes["expected_loss_treatment"],
                        "Recommendation": "Preferred"
                        if bayes["expected_loss_treatment"] < bayes["expected_loss_control"]
                        else "Higher risk",
                    },
                    {
                        "Decision": "If we keep control",
                        "Expected loss rate": bayes["expected_loss_control"],
                        "Recommendation": "Preferred"
                        if bayes["expected_loss_control"] < bayes["expected_loss_treatment"]
                        else "Higher risk",
                    },
                ]
            )
            st.dataframe(loss_df, use_container_width=True)

    if include_segments:
        st.header("Section 4: Segment Analysis")
        seg = results["segment_analysis"].copy()
        left, right = st.columns(2)
        day = seg.loc[seg["segment_type"] == "day_of_week"].copy()
        time = seg.loc[seg["segment_type"] == "time_of_day"].copy()
        with left:
            day_long = day.melt(id_vars=["segment", "sample_size"], value_vars=["control_rate", "treatment_rate"], var_name="group", value_name="rate")
            fig_day = px.bar(
                day_long,
                x="segment",
                y="rate",
                color="group",
                barmode="group",
                color_discrete_map={"control_rate": GREY, "treatment_rate": PRIMARY},
                title="Conversion Rate by Day of Week",
                template="plotly_white",
            )
            fig_day.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_day, use_container_width=True)
            if (day["lift"] < 0).any():
                st.caption("Some days show reversed results (treatment < control).")
        with right:
            time_long = time.melt(id_vars=["segment", "sample_size"], value_vars=["control_rate", "treatment_rate"], var_name="group", value_name="rate")
            fig_time = px.bar(
                time_long,
                x="segment",
                y="rate",
                color="group",
                barmode="group",
                color_discrete_map={"control_rate": GREY, "treatment_rate": PRIMARY},
                title="Conversion Rate by Time of Day",
                template="plotly_white",
            )
            fig_time.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_time, use_container_width=True)

        seg_summary = seg.copy()
        seg_summary["Lift %"] = seg_summary["lift"] * 100
        seg_summary["Notable?"] = seg_summary["Lift %"].apply(
            lambda v: "High positive" if v > 5 else ("High negative" if v < -5 else "")
        )
        seg_summary["Segment"] = seg_summary["segment_type"] + " · " + seg_summary["segment"]
        seg_summary = seg_summary.sort_values(by="Lift %", key=lambda s: s.abs(), ascending=False)
        st.dataframe(
            seg_summary[["Segment", "control_rate", "treatment_rate", "Lift %", "sample_size", "Notable?"]],
            use_container_width=True,
        )

    st.header("Section 5: Recommendation")
    rec_text, next_steps = recommendation_text(results)
    st.info(rec_text)
    st.markdown("**What to do next**")
    for step in next_steps:
        st.write(f"- {step}")

    st.markdown(
        "Project 4 of 6 · Data & Business Analyst Portfolio · "
        "Data: E-Commerce A/B Testing Dataset (Kaggle) · "
        "[View source on GitHub](https://github.com/KOBBY-IM/ab-test-analyser)"
    )


if __name__ == "__main__":
    build_app()
