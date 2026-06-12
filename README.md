# Marketing Mix Modelling — Budget Optimisation & Revenue Attribution

Two years of weekly marketing spend data. One econometric model. $461,310 in additional annual revenue from the same total budget, with no extra spend required.

This project builds a complete Marketing Mix Model for an e-commerce brand from the ground up — adstock transformation, OLS regression with full diagnostics, Ridge regression for production scoring, diminishing returns curves, constrained budget optimisation and a three-scenario what-if analysis. Every step is documented and reproducible.

---

## The problem this solves

The marketing team was spending $19,049 per week across four channels. 54% of that — $10,281 — was going to Paid Search. 4.5% — $855 — was going to Email.

The Email ROAS calculated by this model: **44.10x**.
The Paid Search ROAS: **3.00x**.

That inversion — the most underfunded channel delivering 14x the return of the most overfunded one — is exactly the kind of thing a standard analytics dashboard does not surface. It takes a model that separates each channel's independent contribution, accounts for carryover effects and corrects for seasonality to see it clearly.

The optimised allocation redirects $2,145 per week from Paid Search and Paid Social (both approaching saturation) to Email and Influencer (both with significant headroom). Same $19,049 total. Projected weekly revenue lift: $8,871. Annualised: $461,310.

---

## Dataset

- **104 weeks** — January 2023 through December 2024
- **4 paid channels:** Paid Search (Google), Paid Social (Facebook/Instagram), Email, Influencer/Affiliate
- **Plus:** SEO/Organic index, orders, average order value, new visitors, conversion rate
- **Total revenue modelled:** $20,222,811 across the two-year period
- **Total spend modelled:** $1,981,072
- **Overall blended ROAS:** 11.52x

The dataset was clean: zero missing values, zero duplicates, zero negative spend. The one anomaly flagged was a legitimate Black Friday/Christmas surge in Paid Search (week of 2023-12-18 at $12,729 vs weekly average of $10,281). Influencer had 39 zero-spend weeks — expected for a channel that runs in concentrated campaign bursts rather than continuously.

---

## What the model produces

**Channel revenue attribution (2-year totals):**

| Channel | Revenue Attributed | Share | Total Spend | ROAS |
|---|---|---|---|---|
| Baseline (organic / brand equity) | $7,036,014 | 34.8% | — | — |
| Paid Social | $4,920,112 | 24.3% | $688,738 | 7.14x |
| Email | $3,921,884 | 19.4% | $88,926 | **44.10x** |
| Paid Search | $3,202,438 | 15.8% | $1,069,254 | 3.00x |
| SEO / Organic | $591,778 | 2.9% | $0 | N/A |
| Influencer | $550,585 | 2.7% | $134,154 | 4.10x |

**Optimised budget (same $19,049 weekly):**

| Channel | Current | Optimal | Change |
|---|---|---|---|
| Paid Search | $10,281 | $8,522 | -$1,760 |
| Paid Social | $6,622 | $5,140 | -$1,483 |
| Email | $855 | $3,000 | **+$2,145** |
| Influencer | $1,290 | $2,387 | +$1,098 |

**Weekly revenue:** $104,474 → $113,346 (+8.5%)
**Annual uplift:** +$461,310

---

## Model performance

| Metric | Value | Interpretation |
|---|---|---|
| R-squared | 0.9567 | Model explains 95.7% of revenue variance |
| Adjusted R-squared | 0.9498 | Stays close — no overfitting on redundant features |
| MAE | $3,758 / week | Average weekly prediction error |
| MAPE | 2.0% | Industry benchmark is <15%. This is well inside it. |
| CV R² (5-fold) | 0.761 ± 0.144 | Lower than in-sample — see note below |

**On the CV gap:** The 20-point difference between in-sample R² (0.957) and CV R² (0.761) is partly a methodology artefact. Standard k-fold cross-validation applied to time-series data lets the model train on late 2024 to predict early 2023 — which is not how it will be used in production. The large Black Friday and Christmas patterns also land differently across folds, driving the high variance (±0.144). Time-series cross-validation would give a more honest estimate. Budget decisions from this model should be treated as directional rather than precise.

---

## Notebook structure

| Section | Content |
|---|---|
| 1 | Executive summary and business context |
| 2 | Imports and setup |
| 3 | Data loading, quality report, descriptive statistics |
| 4 | Exploratory analysis — spend trends, correlation matrix, spend vs revenue scatter |
| 5 | Adstock transformation — geometric decay applied to each channel |
| 6 | Seasonality decomposition — Fourier terms and event dummies |
| 7 | OLS model with full statsmodels summary, Ridge model for scoring |
| 8 | Channel contribution decomposition |
| 9 | ROAS by channel |
| 10 | Diminishing returns curves (Hill function) with saturation point analysis |
| 11 | Constrained budget optimisation (SLSQP) |
| 12 | Three-scenario what-if analysis |
| 13 | Executive dashboard |
| 14 | Channel strategy and seasonal budget recommendations |
| 15 | Export — all datasets, model pkl, visuals |

---

## The adstock transformation

Raw spend data has a timing problem. When $7,000 goes into Influencer in a single week, the revenue effect does not all arrive in those seven days — the content stays live, gets reshared and converts viewers days or weeks later. Modelling that $7,000 as a one-week event understates the channel's contribution.

Adstock solves this by converting weekly spend into an effective exposure figure that decays exponentially forward in time:

```python
def adstock_transform(spend_series, decay):
    result = np.zeros(len(spend_series))
    result[0] = spend_series[0]
    for t in range(1, len(spend_series)):
        result[t] = spend_series[t] + decay * result[t-1]
    return result
```

Decay rates used:

| Channel | Decay | Rationale |
|---|---|---|
| Paid Search | 0.35 | Intent-driven — clicks stop when ads stop |
| Paid Social | 0.45 | Awareness lingers for a few weeks |
| Email | 0.20 | Opens happen fast or not at all |
| Influencer | 0.55 | Content circulates long after campaign ends |

These rates are calibrated to known channel mechanics, not measured from holdout experiments. Validating them with geo holdouts is the recommended next step.

---

## Project structure

```
marketing-mix-modelling/
│
├── MMM_Analysis.ipynb                    ← Main notebook (15 sections)
│
├── data/
│   ├── raw/
│   │   └── mmm_ecommerce_dataset.csv     ← 104-week dataset (19 columns)
│   └── processed/
│       ├── mmm_cleaned_data.csv          ← With all engineered features (33 columns)
│       ├── mmm_powerbi.csv               ← Weekly contributions + ROAS per channel (30 cols)
│       ├── budget_optimisation_results.csv
│       └── channel_roi_summary.csv
│
├── models/
│   └── ridge_mmm_model.pkl               ← Trained Ridge model for scoring
│
├── visuals/
│   ├── spend_trend.png
│   ├── spend_distribution.png
│   ├── correlation_matrix.png
│   ├── spend_vs_revenue_scatter.png
│   ├── adstock_transformation.png
│   ├── seasonality_analysis.png
│   ├── model_fit.png
│   ├── channel_contribution.png
│   ├── roi_by_channel.png
│   ├── diminishing_returns.png
│   ├── budget_optimisation.png
│   ├── scenario_analysis.png
│   └── executive_dashboard.png
│
└── README.md
```

---

## How to run it

```bash
git clone https://github.com/anonopatience/marketing-mix-modelling.git
cd marketing-mix-modelling

pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
jupyter notebook MMM_Analysis.ipynb
```

Run Kernel → Restart & Run All. The notebook takes about 30–60 seconds — the optimisation loop runs multiple iterations.

---

## Honest caveats

**The adstock decay rates are assumptions.** They are calibrated to how each channel type generally works. If the true Influencer decay for this specific business is 0.35 rather than 0.55, its attributed contribution would be lower and its ROAS would fall. The only way to validate these values is holdout experiments — pausing a channel for a subset of markets or time periods and measuring the revenue impact directly.

**The saturation curves are modelled, not measured.** The Hill function parameters are fitted to the observed spend range. Extrapolating far beyond that range — doubling Paid Search to $20,000/week, for example — reaches territory where the model is genuinely uncertain.

**Paid Search significance.** The OLS regression shows Paid Search with p = 0.107, which fails the 95% confidence threshold. This does not mean Paid Search does not work — it means the model cannot cleanly separate its independent effect from correlated channel movements. The condition number (9.81e+05) flags multicollinearity. Ridge regression addresses this in the production model, but the ROAS figures should be interpreted with some caution.

**Creative quality is unmodelled.** The model treats all Paid Social spend as equivalent. A great creative week looks identical to a poor one in the data. Average ROAS across varying creative quality is what the model captures.

---

## About

**Patience Anono** — Data Analyst & Marketing Analytics Specialist

📧 anonopatience@gmail.com  
🌐 [padataanalytics.com](https://padataanalytics.com)  
💼 [LinkedIn](https://www.linkedin.com/in/patience-anono-22ab06176/)

---

*Dataset is synthetic, built to mirror real e-commerce marketing data. All methodology — adstock, OLS, Ridge, Hill saturation curves, SLSQP optimisation — is real and production-applicable.*
