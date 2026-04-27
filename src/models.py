"""
Uplift models for churn prediction.

Three meta-learner approaches, all using XGBoost as the base estimator:

  S-Learner  — single model; treatment is just another feature.
  T-Learner  — separate models for treatment and control groups.
  X-Learner  — cross-fitted CATE estimates; best for imbalanced T/C ratios.

Each class wraps the corresponding CausalML implementation and exposes a
uniform interface: fit(), predict(), name.

MLflow tracking
---------------
Call log_all_experiments() after evaluation to persist one MLflow run per model.
Runs are written to ./mlruns by default. Pass tracking_uri to override.
"""

import numpy as np
import pandas as pd

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

from causalml.inference.meta import (
    BaseSClassifier,
    BaseTClassifier,
    BaseXClassifier,
)
from xgboost import XGBClassifier, XGBRegressor


_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

# Default business constants (used to compute optimal threshold)
DEFAULT_CONTACT_COST = 5.0
DEFAULT_CHURN_REVENUE = 200.0


def _xgb():
    return XGBClassifier(**_XGB_PARAMS)


class SLearner:
    name = "S-Learner"

    def __init__(self):
        self._model = BaseSClassifier(learner=_xgb())

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series):
        self._model.fit(X=X, treatment=w, y=y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._model.predict(X=X)
        if hasattr(preds, "shape") and preds.ndim == 2:
            preds = preds.flatten()
        return np.asarray(preds, dtype=float)


class TLearner:
    name = "T-Learner"

    def __init__(self):
        self._model = BaseTClassifier(learner=_xgb())

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series):
        self._model.fit(X=X, treatment=w, y=y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._model.predict(X=X)
        if hasattr(preds, "shape") and preds.ndim == 2:
            preds = preds.flatten()
        return np.asarray(preds, dtype=float)


def _xgb_reg():
    params = {k: v for k, v in _XGB_PARAMS.items() if k != "eval_metric"}
    return XGBRegressor(**params, eval_metric="rmse")


class XLearner:
    name = "X-Learner"

    def __init__(self):
        self._model = BaseXClassifier(
            outcome_learner=_xgb(),
            effect_learner=_xgb_reg(),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series):
        self._model.fit(X=X, treatment=w, y=y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._model.predict(X=X)
        if hasattr(preds, "shape") and preds.ndim == 2:
            preds = preds.flatten()
        return np.asarray(preds, dtype=float)


def train_all(X_train, y_train, w_train):
    """Fit all three models and return a list of fitted model objects."""
    models = [SLearner(), TLearner(), XLearner()]
    for m in models:
        print(f"Training {m.name} ...")
        m.fit(X_train, y_train, w_train)
        print(f"  {m.name} done.")
    return models


def predict_all(models, X_test):
    """Return dict {model_name: cate_array} for every model."""
    return {m.name: m.predict(X_test) for m in models}


# ---------------------------------------------------------------------------
# Optimal threshold
# ---------------------------------------------------------------------------

def optimal_threshold(
    contact_cost: float = DEFAULT_CONTACT_COST,
    churn_revenue: float = DEFAULT_CHURN_REVENUE,
) -> float:
    """
    CATE break-even point: target customers whose predicted uplift exceeds this.

    Derivation: marginal revenue (cate × churn_revenue) > contact_cost
                → cate > contact_cost / churn_revenue
    """
    return contact_cost / churn_revenue


def optimal_targeting_fraction(
    cate_array: np.ndarray,
    contact_cost: float = DEFAULT_CONTACT_COST,
    churn_revenue: float = DEFAULT_CHURN_REVENUE,
) -> float:
    """Fraction of the population with CATE above the break-even threshold."""
    thresh = optimal_threshold(contact_cost, churn_revenue)
    return float(np.mean(cate_array > thresh))


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_experiment(
    model_name: str,
    auuc: float,
    cate_array: np.ndarray,
    contact_cost: float = DEFAULT_CONTACT_COST,
    churn_revenue: float = DEFAULT_CHURN_REVENUE,
    tracking_uri: str | None = None,
    experiment_name: str = "uplift_churn",
) -> str | None:
    """
    Log one MLflow run for a single model.

    Parameters
    ----------
    model_name : str
        e.g. "S-Learner"
    auuc : float
        AUUC score from evaluate.plot_qini_curves()
    cate_array : np.ndarray
        Predicted CATE values on the test set (used to compute threshold fraction)
    tracking_uri : str, optional
        MLflow tracking URI; defaults to ./mlruns
    experiment_name : str
        MLflow experiment name

    Returns
    -------
    run_id : str or None
        The MLflow run ID, or None if mlflow is unavailable.
    """
    if not _MLFLOW:
        print("mlflow not installed — skipping experiment logging.")
        return None

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    thresh = optimal_threshold(contact_cost, churn_revenue)
    frac = optimal_targeting_fraction(cate_array, contact_cost, churn_revenue)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("base_learner", "XGBClassifier")
        # XGB hyperparameters
        for k, v in _XGB_PARAMS.items():
            mlflow.log_param(k, v)
        # Business constants
        mlflow.log_param("contact_cost", contact_cost)
        mlflow.log_param("churn_revenue", churn_revenue)
        # Evaluation metrics
        mlflow.log_metric("auuc", auuc)
        mlflow.log_metric("optimal_cate_threshold", thresh)
        mlflow.log_metric("optimal_targeting_fraction", frac)

        print(
            f"MLflow run logged: {model_name} | "
            f"AUUC={auuc:.4f} | threshold={thresh:.4f} | "
            f"targeting_fraction={frac:.2%} | run_id={run.info.run_id}"
        )
        return run.info.run_id


def log_all_experiments(
    models,
    auuc_scores: dict,
    cate_preds: dict,
    contact_cost: float = DEFAULT_CONTACT_COST,
    churn_revenue: float = DEFAULT_CHURN_REVENUE,
    tracking_uri: str | None = None,
) -> dict:
    """
    Log one MLflow run per model. Returns {model_name: run_id}.

    Called from the notebook after evaluation is complete.
    """
    run_ids = {}
    for m in models:
        rid = log_experiment(
            model_name=m.name,
            auuc=auuc_scores.get(m.name, float("nan")),
            cate_array=cate_preds[m.name],
            contact_cost=contact_cost,
            churn_revenue=churn_revenue,
            tracking_uri=tracking_uri,
        )
        run_ids[m.name] = rid
    return run_ids
