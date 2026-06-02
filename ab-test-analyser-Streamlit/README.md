# A/B Test Analyser

## What it does
End-to-end A/B test analysis tool — from raw experiment data to a statistically grounded, plain-English recommendation.

## Architecture
```text
ab_data.csv
    │
    ▼
[EXTRACT]   load raw experiment data (294,478 rows)
    │
    ▼
[VALIDATE]  detect mismatches, remove duplicates, flag anomalies
    │
    ▼
[TRANSFORM] compute conversion rates, segment features
    │
    ├──→ Frequentist z-test (p-value, CI, power)
    ├──→ Bayesian Beta-Binomial simulation
    ├──→ Effect size (Cohen's h)
    └──→ Segment analysis (by day, time of day)
                  │
                  ▼
         Streamlit Dashboard
    (Verdict · Posteriors · Segments · Recommendation)
```

## Statistical approach
This tool implements two complementary approaches:

Frequentist (z-test):  
The industry standard for A/B testing. Tests whether the observed difference in conversion rates is unlikely to have occurred by chance. Reports p-value, confidence intervals, and statistical power.

Bayesian (Beta-Binomial):  
Models conversion rates as probability distributions rather than point estimates. Reports the probability that treatment outperforms control, and the expected loss from each decision.

Why both? Frequentist testing answers "is this difference real?" Bayesian testing answers "what should we do?" They complement each other and disagreement between them is itself an analytical signal.

## Key analytical decision: mismatch removal
The Kaggle dataset contains rows where control users were shown the new page and treatment users were shown the old page. These are experimental contamination errors — including them would bias the analysis. The ETL pipeline detects and removes all mismatched rows before any statistics are run, and documents the count and percentage removed. This decision is flagged in the data quality report.

## Quickstart
```bash
pip install -r requirements.txt
# Download dataset to data/raw/ab_data.csv
streamlit run app.py
```

## Run analysis from CLI
```bash
python src/etl.py --input data/raw/ab_data.csv
python src/stats.py --input data/raw/ab_data.csv --output results.json
```

## Run tests
```bash
pytest tests/ -v
```

## Key findings (from the real dataset)
- After mismatch and duplicate removal: ~286,690 clean users
- Control conversion rate: ~12.0%
- Treatment conversion rate: ~11.9%
- Result: no statistically significant difference (p ≈ 0.19)
- Interpretation: the new page does not improve conversions — do not deploy based on this experiment alone
- Bayesian: P(treatment better) ≈ 41% — favours control

## What makes this project stand out
- Both frequentist AND Bayesian — most analysts only use one
- Mismatch detection — shows data quality thinking before modelling
- Plain-English verdict — bridges statistical output and business decision
- Power analysis — tells you whether your test was even large enough to detect the effect you were looking for
- Segment analysis — checks whether the result holds across days/times

## Extending this project
- Add sequential testing (always-valid p-values) for continuous monitoring
- Add multi-variant support (A/B/C testing, not just A/B)
- Add CUPED variance reduction (pre-experiment covariate adjustment)
- Connect to a live experiment platform API (Optimizely, LaunchDarkly)
- Add a sample size calculator as a pre-experiment planning tool

## Data source
E-Commerce A/B Testing Dataset — 294,478 rows, 5 columns.  
Kaggle: [kaggle.com/datasets/zhangluyuan/ab-testing](https://kaggle.com/datasets/zhangluyuan/ab-testing)  
Alternative: [github.com/baumanab/udacity_ABTesting](https://github.com/baumanab/udacity_ABTesting)

## Tech stack
| Tool | Role |
|---|---|
| Python / pandas | ETL and data preparation |
| scipy | Two-proportion z-test |
| statsmodels | Power analysis |
| numpy | Bayesian simulation |
| plotly | Interactive charts |
| Streamlit | Dashboard and UI |
| pytest | Unit tests |

## Author
Theophilus Nyarko-Mensah · [LinkedIn](https://www.linkedin.com/in/theophilus-nyarko-mensah-8b2063129)
