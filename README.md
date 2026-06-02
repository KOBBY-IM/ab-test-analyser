# A/B Test Analyser

End-to-end A/B test analysis tool built in two implementations: a full-stack Python/Streamlit dashboard and a standalone static HTML app.

---

## Implementations

### Streamlit App — `ab-test-analyser- Streamlit/`

Python-powered dashboard for deep analysis of the Udacity A/B testing dataset (294,478 rows).

**Stack:** Python · pandas · scipy · numpy · plotly · Streamlit · pytest

**Quickstart:**
```bash
pip install -r "ab-test-analyser- Streamlit/requirements.txt"
# Place ab_data.csv in ab-test-analyser- Streamlit/data/raw/
streamlit run "ab-test-analyser- Streamlit/app.py"
```

**Structure:**
```
ab-test-analyser- Streamlit/
├── app.py                  # Streamlit dashboard entry point
├── src/
│   ├── etl.py              # Extract, validate, transform pipeline
│   └── stats.py            # z-test, Bayesian simulation, effect size
├── tests/
│   └── test_stats.py       # pytest unit tests
├── data/raw/               # Place ab_data.csv here
└── requirements.txt
```

**Features:**
- ETL pipeline — mismatch detection, deduplication, data quality report
- Frequentist z-test — p-value, 95% confidence intervals, observed power
- Bayesian Beta-Binomial — posterior distributions, P(treatment > control), expected loss
- Cohen's h effect size
- Segment analysis — conversion rates by day and time of day
- Plain-English verdict and deployment recommendation

**Key findings (real dataset):**
- Clean rows after ETL: ~286,690 users
- Control: ~12.0% conversion · Treatment: ~11.9% conversion
- p ≈ 0.19 — no statistically significant difference
- P(treatment better) ≈ 41% — Bayesian evidence favours control
- Verdict: do not deploy the new page based on this experiment

---

### Static HTML App — `ab-test-analyser-static/`

Zero-dependency single-file app. No server, no build tools, no Python. Open in any browser.

**Stack:** HTML · CSS · Vanilla JS · Chart.js 4.4.1 (CDN) · PapaParse 5.4.1 (CDN)

**Usage:**
```
Open ab-test-analyser-static/ab-test-analyser.html in a browser
```

**Structure:**
```
ab-test-analyser-static/
└── ab-test-analyser.html   # ~1,700 lines — all CSS, HTML, JS inline
```

**Features:**
- CSV upload (drag-and-drop) or built-in 500-row sample data
- Animated 4-step ETL pipeline (Extract → Validate → Transform → Ready)
- Data quality cards — raw rows, mismatches removed, duplicates removed, clean rows
- Frequentist tab — z-test, CI error bar chart, p-value visual, Cohen's h, power bar
- Bayesian tab — 50,000 Monte Carlo samples, posterior Beta curves, SVG probability gauge, expected loss table
- AI Interpretation — optional Claude API integration (user supplies key; only summary metrics sent)
- Executive recommendation memo — deploy / continue testing / do not deploy

---

## Statistical Approach

Both implementations use the same two-method framework:

**Frequentist (two-proportion z-test)**
Tests whether the observed conversion rate difference is unlikely under the null hypothesis. Reports p-value (α = 0.05), 95% confidence intervals, Cohen's h effect size, and observed power.

**Bayesian (Beta-Binomial)**
Models conversion rates as probability distributions using conjugate Beta priors. Reports P(treatment > control) via Monte Carlo simulation and expected loss under each decision.

**Why both?** Frequentist answers *"is this difference real?"* Bayesian answers *"what should we do?"* Disagreement between them is itself a signal worth investigating.

---

## Dataset

Udacity / Kaggle A/B Testing Dataset — 294,478 rows, 5 columns.

| Column | Values |
|---|---|
| `user_id` | Unique user identifier |
| `timestamp` | Event timestamp |
| `group` | `control` or `treatment` |
| `landing_page` | `old_page` or `new_page` |
| `converted` | `0` or `1` |

**Mismatch removal:** rows where a control user saw the new page or a treatment user saw the old page are experimental contamination and are removed before any statistics are run.

---

## Author

Theophilus Nyarko-Mensah · [LinkedIn](https://www.linkedin.com/in/theophilus-nyarko-mensah-8b2063129)


