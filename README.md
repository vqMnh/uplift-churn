# Uplift Modeling for Customer Churn Prediction

A portfolio project demonstrating **uplift modeling** (heterogeneous treatment-effect
estimation) applied to customer churn. The goal is to identify which customers will
stop churning *because* of a retention offer — not just which customers are most at risk.

---

## Business Problem

A telecom company spends budget on retention discounts (e.g. bill credits, free upgrades).
Two groups of customers are unhelpful to target:

| Group | Problem |
|-------|---------|
| **"Sure things"** — would have stayed anyway | Wastes retention budget |
| **"Lost causes"** — will churn regardless | Budget has zero effect |

Uplift models find the **"persuadables"** — customers whose retention probability
genuinely increases when contacted. This maximises return on campaign spend.

---

## Dataset

**IBM Telco Customer Churn** — 7,043 rows, 21 columns. Publicly available at:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

### Synthetic Treatment Column

> **Important assumption:** The public dataset has no randomised treatment assignment.

We simulate a binary column `received_retention_offer` (1 = offered a discount,
0 = control) with a **40 % treatment probability**, drawn **independently of all
features**. This creates a valid randomised-controlled-trial (RCT) analogue for
training and evaluation purposes.

In a production setting, this column would come from your CRM or A/B testing
infrastructure. All conclusions about CATE magnitudes should be interpreted in the
context of this simulation.

---

## Methodology

### Meta-Learner Framework

| Model | How it works | Strengths |
|-------|-------------|-----------|
| **S-Learner** | One model; treatment flag is just another input feature | Simple baseline |
| **T-Learner** | Separate models for treated and control; CATE = μ₁(x) − μ₀(x) | Captures heterogeneity |
| **X-Learner** | Cross-fits imputed treatment effects; weighted by propensity | Best for imbalanced T/C |

All base estimators: **XGBoost** (n_estimators=300, max_depth=5, lr=0.05).

### Evaluation: Qini Curves

A **Qini curve** answers: *"If I contact the top k% of customers ranked by predicted
uplift, how many churn events do I prevent above random targeting?"*

- Higher, steeper curve = better uplift model
- **AUUC** summarises the curve as a single number (0.5 = random, 1.0 = perfect)

**Non-technical interpretation:** With a budget to call 200 of 1,000 customers, a random
list saves X churns. A good uplift model's list saves significantly more. The Qini curve
shows how much better at every possible budget level.

### Business ROI

Break-even CATE threshold = contact_cost / churn_revenue = $5 / $200 = **0.025**.
Only target customers whose predicted uplift exceeds this threshold.

```
Net Lift = (avg_CATE × customers_targeted × $200) − (customers_targeted × $5)
```

---

## Project Structure

```
uplift_churn/
├── data/
│   ├── raw/                  downloaded CSV (auto-created on first run)
│   └── processed/            reserved for future use
├── src/
│   ├── ingest.py             downloads / caches the Telco dataset
│   ├── preprocess.py         cleans, engineers features, adds synthetic treatment
│   ├── models.py             S/T/X-Learner wrappers + MLflow tracking
│   └── evaluate.py           Qini curves, SHAP, ROI table, fairness report
├── notebooks/
│   ├── colab_pipeline.ipynb  full pipeline (train → evaluate → MLflow)
│   └── colab_dashboard.ipynb interactive dashboard (Plotly + ipywidgets)
├── outputs/
│   └── figures/              all saved plots + CSV outputs
├── tests/
│   └── test_pipeline.py      pytest smoke tests
├── model_card.md             intended use, limitations, fairness considerations
├── requirements.txt          pinned dependencies (runtime + dev)
└── setup_venv.sh             creates .venv and installs everything
```

---

## Quickstart

### Google Colab (recommended)

1. Open `notebooks/colab_pipeline.ipynb` in Colab
2. Update `GITHUB_URL` in the first cell to point to your fork
3. Run all cells — outputs go to `Google Drive/churn_project/`
4. Open `notebooks/colab_dashboard.ipynb` for the interactive dashboard

### Local

```bash
git clone https://github.com/YOUR_USERNAME/uplift_churn.git
cd uplift_churn
bash setup_venv.sh
source .venv/bin/activate
jupyter notebook notebooks/colab_pipeline.ipynb
```

### Run tests

```bash
pytest tests/ -v
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/figures/qini_curves.png` | Qini curves for all three models |
| `outputs/figures/shap_summary.png` | SHAP bar chart (best model) |
| `outputs/figures/shap_beeswarm.png` | SHAP beeswarm (best model) |
| `outputs/figures/roi_net_lift.png` | Net lift bar chart |
| `outputs/roi_table.csv` | Full ROI numbers |
| `outputs/fairness_report.csv` | Mean CATE by gender and senior-citizen subgroup |
| `mlruns/` | MLflow experiment history |

---

## Assumptions and Limitations

1. **Synthetic treatment**: 40 % RCT simulation — not real campaign data
2. **Label encoding**: ordinal codes for nominal features; tree models compensate partially
3. **No propensity calibration** for X-Learner in observational settings
4. **$5 / $200 business constants** are illustrative — replace with actuals

See `model_card.md` for full fairness analysis and ethical considerations.

---

## Dependencies

See `requirements.txt`. Core libraries: CausalML · XGBoost · SHAP · scikit-learn ·
pandas · numpy · matplotlib · Plotly · ipywidgets · MLflow · pytest
