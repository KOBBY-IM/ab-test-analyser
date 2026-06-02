# A/B Test Analyser

A single-file HTML application for statistical analysis of A/B test experiments. Upload experiment data or load sample data to get frequentist z-tests, Bayesian simulations, confidence intervals, effect sizes, and AI-powered business recommendations.

## Features

- **Data Quality Report** — ETL pipeline with validation, mismatch detection, and duplicate removal
- **Experiment Overview** — KPI cards showing conversion rates, absolute/relative lift, and comparative chart
- **Frequentist Analysis** — Two-proportion z-test, 95% confidence intervals, Cohen's h effect size, observed power
- **Bayesian Analysis** — 50,000 Monte Carlo simulations, posterior distributions, probability gauge, expected loss table
- **AI Interpretation** — Optional Claude AI summaries (user provides API key; data never sent, only summary metrics)
- **Executive Recommendation** — Plain-English deployment guidance with next steps
- **Fully Responsive** — Desktop & mobile optimized; works on all modern browsers

## Usage

### Load Data
1. **Upload CSV** — Drag `ab_data.csv` or click to browse. Expected columns: `user_id`, `timestamp`, `group`, `landing_page`, `converted`
2. **Load Sample Data** — Click button to populate with 500 synthetic rows (250 control, 250 treatment, 10 mismatches)

### View Results
- **Sections** — Data Quality → Overview → Frequentist/Bayesian Stats → AI Interpretation → Recommendation
- **Tabs** — Switch between Frequentist and Bayesian analysis without page reload
- **Charts** — Chart.js visualizations (bar chart, confidence intervals, posterior curves, probability gauge)

### Optional: Claude AI Interpretation
1. Enter your Anthropic API key (from [console.anthropic.com](https://console.anthropic.com))
2. Click "Generate Interpretation"
3. Claude reads your summary statistics and returns a business-focused verdict (no raw data sent)

## Technical

- **Single File** — `ab-test-analyser.html` (~1,700 lines; all CSS, HTML, JS inline)
- **No Build Tools** — Pure vanilla JavaScript; no frameworks, npm, or Python
- **CDN Libraries**
  - Chart.js 4.4.1 (charting)
  - PapaParse 5.4.1 (CSV parsing)
  - Google Fonts (DM Serif Display, DM Sans)
- **Colour System** — Matches portfolio design (ink, paper, accent blue, semantic status colours)
- **Statistical Formulas**
  - Two-proportion z-test with pooled proportion
  - Normal CDF via Abramowitz & Stegun approximation
  - Beta PDF via Lanczos log-gamma
  - Gamma sampling via Marsaglia & Tsang
  - Cohen's h for effect size
  - Observed power calculation
- **Browser Support** — Modern browsers (Chrome, Firefox, Safari, Edge)

## Dataset Format

**Required columns:**
- `user_id` — unique identifier
- `timestamp` — ISO 8601 or similar (for display only)
- `group` — "control" or "treatment"
- `landing_page` — "old_page" or "new_page"
- `converted` — 0 or 1

**Data cleaning:**
- Rows where control users saw new_page or treatment users saw old_page are flagged as mismatches and removed
- Duplicate user_ids are deduplicated (first occurrence kept)

## Example

```
user_id,timestamp,group,landing_page,converted
10001,2024-01-01 08:32:15,control,old_page,0
10002,2024-01-01 09:14:22,treatment,new_page,1
10003,2024-01-01 10:05:33,control,old_page,0
...
```

## Interpretation Guide

### Frequentist Verdict
- **Green (p < 0.05, treatment > control)** — Deploy the new page
- **Red (p < 0.05, treatment < control)** — Do not deploy; investigate variant
- **Amber (p ≥ 0.05)** — No significant difference; continue testing

### Bayesian Verdict
- **P(treatment better) > 95%** — Strong evidence to deploy
- **P(treatment better) > 80%** — Moderate evidence to deploy
- **P(treatment better) ≤ 50%** — Evidence favours control

### Effect Size (Cohen's h)
- **Negligible** (|h| < 0.2) — Statistical significance ≠ practical impact
- **Small** (|h| < 0.5) — Minor effect
- **Medium** (|h| < 0.8) — Moderate effect
- **Large** (|h| ≥ 0.8) — Substantial effect

## Notes

- API key is never stored or sent except directly to Anthropic's API (in-browser only)
- Zero conversions in either group → warning shown; statistics still computed but interpret with caution
- Expected power < 80% → test may be underpowered; run longer or redesign variant
- Overlapping 95% CIs → inconclusive; insufficient evidence for deployment

---

