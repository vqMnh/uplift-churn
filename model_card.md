# Model Card — Uplift Models for Customer Churn Prevention

*Following the Model Cards for Model Reporting framework (Mitchell et al., 2019)*

---

## Model Details

| Field | Value |
|-------|-------|
| Model family | Meta-learner uplift / CATE estimation |
| Variants | S-Learner, T-Learner, X-Learner |
| Base estimator | XGBoost classifier (n_estimators=300, max_depth=5, lr=0.05) |
| Outcome | P(churn prevented \| treated) − P(churn prevented \| not treated) per customer |
| Version | 1.0 — portfolio demonstration |
| Developed by | Portfolio project; not affiliated with IBM or the dataset authors |
| License | MIT |

---

## Intended Use

### Primary intended use

Rank customers by their **individual treatment effect** (uplift) so that a
retention campaign can concentrate budget on **persuadable** customers —
those who would stop churning *because* they received the offer.

### Intended users

- Customer retention analysts prioritising outreach lists
- Data science teams evaluating meta-learner approaches for marketing mix

### Out-of-scope uses

- **Binary churn prediction**: use a standard classifier instead; this model
  predicts *incremental effect*, not churn probability.
- **Causal inference with observational data**: the model requires a
  randomised or quasi-randomised treatment assignment. Do not apply it to
  data where offer assignment was self-selected or policy-driven without
  adjustment.
- **Real-time scoring at scale**: no latency optimisation has been applied;
  the models are trained for offline batch ranking.

---

## Training Data

### Dataset

**IBM Telco Customer Churn** (public domain)  
URL: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

- **Size**: 7,043 rows × 21 columns
- **Churn rate**: approximately 26.5 % positive class
- **Features**: tenure, monthly/total charges, contract type, payment method,
  internet and phone service flags, demographic attributes (gender, age/senior flag)

### Critical assumption — synthetic treatment

> **The Telco dataset contains no randomised experiment.** The column
> `received_retention_offer` is *synthetically generated* using independent
> Bernoulli draws with p = 0.40, completely independent of all covariates.

This means:

1. The training data is a **simulated RCT**, not real campaign data.
2. Any learned CATE estimates reflect the model's capacity to detect
   *outcome heterogeneity*, not real treatment response.
3. In production, this column must be replaced with actual randomised
   assignment from a CRM A/B test. **Do not deploy this model without
   real experimental data.**

### Pre-processing

- `TotalCharges` coerced to numeric (whitespace → imputed with `MonthlyCharges`)
- `customerID` dropped
- Two engineered features: `charge_per_month`, `is_long_tenure`
- All categorical columns label-encoded (sklearn `LabelEncoder`, alphabetical order)
- 80/20 stratified train/test split (random_state=42)

---

## Evaluation

### Metric: AUUC (Area Under the Uplift Curve)

The Qini-based AUUC measures how much better than random targeting the model
achieves at every possible budget level. CausalML reports a normalised score
in [0, 1] where **0.5 = random** and **1.0 = perfect**.

| Model | Typical AUUC (simulated data) |
|-------|-------------------------------|
| S-Learner | ≈ 0.50–0.55 |
| T-Learner | ≈ 0.50–0.55 |
| X-Learner | ≈ 0.50–0.56 |

> Because treatment is random and independent of covariates, learned CATEs
> reflect noise rather than genuine heterogeneity. AUUC scores on real
> experimental data are expected to be higher.

### Business ROI (illustrative)

Targeting the top 20 % of customers by predicted CATE, with default business
parameters (contact cost = $5, churn value = $200):

| Model | Net Lift | ROI % |
|-------|----------|-------|
| S-Learner | see `outputs/roi_table.csv` | — |
| T-Learner | see `outputs/roi_table.csv` | — |
| X-Learner | see `outputs/roi_table.csv` | — |

*Run `notebooks/colab_pipeline.ipynb` to populate these values.*

---

## Known Limitations

1. **No real treatment signal**: The synthetic treatment means models cannot
   learn genuine heterogeneous treatment effects. AUUC scores on simulated
   data are near-random by construction.

2. **Small dataset**: With ~7k rows and a 40/60 treatment split (~2,800
   treated), the T- and X-Learners have limited data per arm. Confidence
   intervals around individual CATE estimates are wide.

3. **Label encoding**: `LabelEncoder` assigns ordinal codes to nominal
   features (e.g. `InternetService`: DSL=0, Fiber optic=1, No=2). This
   implicitly imposes an ordering that tree models can partially compensate
   for, but one-hot encoding is preferred for linear base learners.

4. **No propensity calibration for X-Learner**: CausalML's `BaseXClassifier`
   uses a simple propensity model internally. For observational data with
   confounders, a separately calibrated propensity model is required.

5. **Static snapshot**: The Telco dataset is a cross-sectional snapshot.
   Temporal dynamics (customer lifecycle, seasonal churn patterns) are not
   captured.

---

## Fairness Considerations

### Protected attributes

Two sensitive attributes are present in the dataset:

| Attribute | Values | Encoding |
|-----------|--------|----------|
| `gender` | Male, Female | LabelEncoder: Female=0, Male=1 |
| `SeniorCitizen` | 0 (< 65), 1 (≥ 65) | Already binary |

### Disparity analysis

We compute the mean predicted CATE for each subgroup. A **disparity ratio**
(max_mean / min_mean) close to 1.0 indicates equitable treatment; values
substantially above 1.0 warrant investigation.

Run `evaluate.fairness_report(X_test, cate_preds)` to generate
`outputs/fairness_report.csv` with the full breakdown.

**Expected findings (simulated data)**:  
Because treatment is simulated independently of features, expected mean CATE
across subgroups should be similar. Deviations > ±10 % of the grand mean
suggest the base learner is inadvertently using a protected attribute as a
proxy for outcome heterogeneity. This can occur even when the feature is not
directly causal, via correlations with other predictors (e.g. `MonthlyCharges`
correlates with `SeniorCitizen` in this dataset).

### Recommended mitigations

- Audit the disparity ratio before deploying targeting lists.
- If `gender` or `SeniorCitizen` is found to be a top SHAP driver,
  consider dropping it from the feature set or applying post-hoc
  calibration per group.
- Consult legal/compliance on whether differential targeting rates by
  demographic group comply with applicable regulations (e.g. ECOA in the US).

---

## Ethical Considerations

- **Selection effects**: Targeting only "persuadables" means some high-risk
  customers in the control group will receive no offer. Operators should decide
  whether a universal safety net (e.g., auto-cancel prevention) is appropriate.
- **Revenue focus**: The ROI metric optimises company revenue, not customer
  welfare. An offer-acceptance that retains a customer who would have preferred
  to leave is a valid ethical concern.
- **Data provenance**: The Telco dataset is a synthetic IBM sample; no real
  customers are represented.

---

## MLflow Experiment Tracking

Experiment runs are stored in `./mlruns/` (or synced to Drive). Each run records:

| Logged item | Key |
|-------------|-----|
| Model type | `model_type` |
| XGB hyperparameters | `n_estimators`, `max_depth`, `learning_rate`, … |
| Business constants | `contact_cost`, `churn_revenue` |
| AUUC score | `auuc` |
| Break-even CATE threshold | `optimal_cate_threshold` |
| Fraction of customers above threshold | `optimal_targeting_fraction` |

View runs: `mlflow ui --backend-store-uri ./mlruns`

---

## Citation

If you use this project as a reference:

```
Uplift Modeling for Customer Churn — Portfolio Project
Dataset: IBM Telco Customer Churn (public domain)
Uplift framework: CausalML (Uber, Apache 2.0)
```
