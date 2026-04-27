"""
Smoke tests for the uplift churn pipeline.

Tests are designed to run quickly (small synthetic fixtures, no network I/O)
and to verify that the preprocessing and evaluation infrastructure produce
correct outputs, not to validate model convergence on real data.

Run with:
    pytest tests/
or from a Colab cell:
    !pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_features, split
from src.models import SLearner, optimal_threshold, optimal_targeting_fraction


def _compute_auuc(cate, y, w) -> float:
    """
    Normalized AUUC: mean Qini gain when customers are ranked by predicted CATE.

    For a random model this is ≈ 0; a positive value means the model beats random.
    Implemented manually so the test is not sensitive to causalml version changes.
    """
    cate = np.asarray(cate, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    order = np.argsort(-cate)
    y, w = y[order], w[order]

    n_treated = w.sum()
    n_control = (1 - w).sum()
    if n_treated == 0 or n_control == 0:
        return 0.0

    cum_t = np.cumsum(w * y) / n_treated
    cum_c = np.cumsum((1 - w) * y) / n_control
    return float((cum_t - cum_c).mean())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_telco_like(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic dataframe that mirrors the Telco column schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.integers(0, 2, size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "tenure": rng.integers(0, 73, size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
            ),
            "MonthlyCharges": rng.uniform(18.0, 118.0, size=n).round(2),
            "TotalCharges": rng.uniform(18.0, 8000.0, size=n).round(2),
            "Churn": rng.choice(["Yes", "No"], size=n, p=[0.265, 0.735]),
        }
    )


def _make_uplift_fixture(n: int = 1_000, seed: int = 0) -> tuple:
    """
    Synthetic dataset with a clear, detectable uplift signal.

    Outcome Y=1 means *retained* (positive event).  Treatment increases
    retention probability by ~10-25 pp (heterogeneous on feature1).
    BaseSClassifier.predict() returns P(Y=1|T=1) - P(Y=1|T=0), so CATE > 0
    for customers who respond — ranking descending is correct for targeting.
    """
    rng = np.random.default_rng(seed)
    feature1 = rng.normal(size=n)
    feature2 = rng.normal(size=n)
    treatment = rng.binomial(1, 0.5, n)

    # Treatment effect is heterogeneous: larger for feature1 > 0
    true_cate = 0.10 + 0.15 * (feature1 > 0).astype(float)
    p_retain_control = 0.60 - 0.10 * (feature1 > 0).astype(float)
    p_retain = p_retain_control + true_cate * treatment
    outcome = rng.binomial(1, p_retain.clip(0.05, 0.95), n)

    X = pd.DataFrame({"feature1": feature1, "feature2": feature2})
    y = pd.Series(outcome, name="outcome")
    w = pd.Series(treatment, name="treatment")
    return X, y, w


# ---------------------------------------------------------------------------
# Test 1: preprocessing produces no nulls
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_no_nulls_after_build_features(self):
        raw = _make_raw_telco_like(n=200)
        processed = build_features(raw)
        null_counts = processed.isnull().sum()
        nulls = null_counts[null_counts > 0]
        assert nulls.empty, f"Nulls found after preprocessing:\n{nulls}"

    def test_treatment_column_binary(self):
        raw = _make_raw_telco_like(n=200)
        processed = build_features(raw)
        assert "received_retention_offer" in processed.columns
        unique_vals = set(processed["received_retention_offer"].unique())
        assert unique_vals == {0, 1}, f"Expected {{0, 1}}, got {unique_vals}"

    def test_split_produces_correct_shapes(self):
        raw = _make_raw_telco_like(n=200)
        processed = build_features(raw)
        X_tr, X_te, y_tr, y_te, w_tr, w_te = split(processed, test_size=0.20)
        n = len(processed)
        assert len(X_tr) + len(X_te) == n
        assert len(y_tr) == len(X_tr)
        assert len(w_tr) == len(X_tr)

    def test_customerid_dropped(self):
        raw = _make_raw_telco_like(n=100)
        processed = build_features(raw)
        assert "customerID" not in processed.columns

    def test_engineered_features_present(self):
        raw = _make_raw_telco_like(n=100)
        processed = build_features(raw)
        assert "charge_per_month" in processed.columns
        assert "is_long_tenure" in processed.columns


# ---------------------------------------------------------------------------
# Test 2: AUUC score above random baseline for best model
# ---------------------------------------------------------------------------

class TestAUUC:
    """
    Train an S-Learner on a synthetic fixture with a clear uplift signal and
    verify the normalized AUUC score is positive (above the random baseline of 0).
    """

    @pytest.fixture(scope="class")
    def trained_results(self):
        X, y, w = _make_uplift_fixture(n=1_000)
        split_idx = 800
        X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
        w_tr, w_te = w.iloc[:split_idx], w.iloc[split_idx:]

        model = SLearner()
        model.fit(X_tr, y_tr, w_tr)
        cate = model.predict(X_te)
        return cate, y_te, w_te

    def test_best_model_auuc_above_random(self, trained_results):
        cate, y_te, w_te = trained_results
        score = _compute_auuc(cate, y_te.values, w_te.values)
        assert score > 0, (
            f"AUUC={score:.4f} is not above the random baseline of 0. "
            "The model may not be learning the uplift signal."
        )

    def test_cate_predictions_finite(self, trained_results):
        cate, _, _ = trained_results
        assert np.all(np.isfinite(cate)), "CATE predictions contain NaN or Inf"

    def test_cate_has_variance(self, trained_results):
        cate, _, _ = trained_results
        assert cate.std() > 1e-6, "CATE predictions are all identical — model may have collapsed"


# ---------------------------------------------------------------------------
# Test 3: optimal threshold helpers
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_optimal_threshold_value(self):
        # contact_cost=5, churn_revenue=200 → threshold=0.025
        thresh = optimal_threshold(contact_cost=5.0, churn_revenue=200.0)
        assert abs(thresh - 0.025) < 1e-9

    def test_optimal_targeting_fraction_range(self):
        X, y, w = _make_uplift_fixture(n=200)
        model = SLearner()
        model.fit(X, y, w)
        cate = model.predict(X)
        frac = optimal_targeting_fraction(cate, contact_cost=5.0, churn_revenue=200.0)
        assert 0.0 <= frac <= 1.0
