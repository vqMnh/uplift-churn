"""
Evaluation utilities for uplift models.

Provides:
  - Qini curve + AUUC score for each model
  - SHAP explanation for the best model (summary plots + raw values)
  - Business ROI table (top 10 / 20 / 30 % targeting)
  - Fairness disparity report across gender and senior citizen groups

All plots are saved to outputs/figures/ automatically, and optionally
mirrored to a Google Drive path when running in Colab.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from causalml.metrics import get_qini, auuc_score

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures"

# Business constants
CONTACT_COST = 5.0
CHURN_REVENUE = 200.0


def _ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, filename: str, drive_dir: pathlib.Path | None = None):
    _ensure_dir(FIGURES_DIR)
    local_path = FIGURES_DIR / filename
    fig.savefig(local_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {local_path}")
    if drive_dir:
        _ensure_dir(drive_dir)
        drive_path = drive_dir / filename
        fig.savefig(drive_path, dpi=150, bbox_inches="tight")
        print(f"Saved to Drive: {drive_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qini curves & AUUC
# ---------------------------------------------------------------------------

def compute_qini_arrays(
    cate_preds: dict,
    y_test: pd.Series,
    w_test: pd.Series,
) -> dict:
    """
    Return {model_name: (x_vals, y_vals)} for Plotly / interactive plotting.

    x_vals — fractions of population targeted [0, 1]
    y_vals — cumulative incremental Qini gains
    """
    result = {}
    y, w = y_test.values, w_test.values
    for name, cate in cate_preds.items():
        df_input = pd.DataFrame({"y": y, "w": w, name: cate})
        qini_df = get_qini(
            df_input,
            outcome_col="y",
            treatment_col="w",
            treatment_effect_col=name,
        )
        x_vals = np.linspace(0, 1, len(qini_df))
        y_vals = qini_df[name].values
        result[name] = (x_vals, y_vals)
    return result


def compute_qini_data(
    cate_preds: dict,
    y_test: pd.Series,
    w_test: pd.Series,
) -> dict:
    """Return {model_name: qini_df} for each model."""
    results = {}
    y, w = y_test.values, w_test.values
    for name, cate in cate_preds.items():
        df_qini = get_qini(
            pd.DataFrame({"y": y, "w": w, name: cate}),
            outcome_col="y",
            treatment_col="w",
            treatment_effect_col=name,
        )
        results[name] = df_qini
    return results


def plot_qini_curves(
    cate_preds: dict,
    y_test: pd.Series,
    w_test: pd.Series,
    drive_dir: pathlib.Path | None = None,
) -> dict:
    """Plot and save Qini curves; return AUUC scores per model."""
    y, w = y_test.values, w_test.values

    scores = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for (name, cate), color in zip(cate_preds.items(), colors):
        df_input = pd.DataFrame({"y": y, "w": w, name: cate})
        qini_df = get_qini(
            df_input,
            outcome_col="y",
            treatment_col="w",
            treatment_effect_col=name,
        )
        auuc = auuc_score(
            df_input,
            outcome_col="y",
            treatment_col="w",
            treatment_effect_col=name,
        )
        scores[name] = auuc

        x_vals = np.linspace(0, 1, len(qini_df))
        ax.plot(x_vals, qini_df[name].values, label=f"{name} (AUUC={auuc:.4f})", color=color, lw=2)

    ax.plot([0, 1], [0, 0], "k--", label="Random baseline", lw=1.5)
    ax.set_xlabel("Proportion of population targeted")
    ax.set_ylabel("Incremental gains (Qini)")
    ax.set_title("Qini Curves — Uplift Models")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save(fig, "qini_curves.png", drive_dir)
    return scores


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_background: int = 200,
):
    """
    Return a SHAP Explanation object for X_test without plotting.

    Used by the dashboard to get per-customer SHAP values for waterfall charts.
    Returns None if the inner estimator cannot be extracted.
    """
    inner = _extract_inner_estimator(model)
    if inner is None:
        return None
    background = shap.sample(X_train, n_background, random_state=42)
    explainer = shap.TreeExplainer(inner, data=background)
    return explainer(X_test)


def explain_best_model(
    best_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drive_dir: pathlib.Path | None = None,
    n_background: int = 200,
):
    """Compute SHAP values, save summary and beeswarm plots."""
    shap_values = compute_shap_values(best_model, X_train, X_test, n_background)
    if shap_values is None:
        print("Could not extract inner estimator for SHAP — skipping.")
        return None

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.gca().set_title(f"SHAP Feature Importance — {best_model.name}")
    _save(fig, "shap_summary.png", drive_dir)

    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    fig2 = plt.gcf()
    fig2.set_size_inches(8, 6)
    plt.gca().set_title(f"SHAP Beeswarm — {best_model.name}")
    _save(fig2, "shap_beeswarm.png", drive_dir)

    return shap_values


def _extract_inner_estimator(model):
    """Walk the CausalML wrapper to reach the fitted XGBClassifier."""
    if hasattr(model._model, "model_"):
        return model._model.model_
    if hasattr(model._model, "model_t_"):
        return model._model.model_t_
    if hasattr(model._model, "models_mu_t_"):
        learners = model._model.models_mu_t_
        if isinstance(learners, list) and learners:
            return learners[0]
    return None


# ---------------------------------------------------------------------------
# Business ROI table
# ---------------------------------------------------------------------------

def compute_roi_curve(
    cate_array: np.ndarray,
    n_total: int,
    contact_cost: float,
    churn_revenue: float,
    n_points: int = 100,
):
    """
    Return (fractions, net_lifts) arrays for the full ROI curve.

    Used by the interactive dashboard slider. fractions ∈ (0, 1].
    """
    order = np.argsort(-cate_array)
    cate_sorted = cate_array[order]
    fractions = np.linspace(0.01, 1.0, n_points)
    net_lifts = np.empty(n_points)
    for i, frac in enumerate(fractions):
        k = max(1, int(frac * n_total))
        mean_cate = float(np.mean(cate_sorted[:k]))
        gross = mean_cate * k * churn_revenue
        cost = k * contact_cost
        net_lifts[i] = gross - cost
    return fractions, net_lifts


def roi_table(
    cate_preds: dict,
    y_test: pd.Series,
    w_test: pd.Series,
    contact_cost: float = CONTACT_COST,
    churn_revenue: float = CHURN_REVENUE,
    percentiles: tuple = (0.10, 0.20, 0.30),
    drive_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Compute expected net lift for each model at target percentiles."""
    rows = []
    n_total = len(y_test)

    for name, cate in cate_preds.items():
        order = np.argsort(-cate)
        cate_sorted = cate[order]

        for pct in percentiles:
            k = max(1, int(pct * n_total))
            top_cate = cate_sorted[:k]
            mean_cate = float(np.mean(top_cate))
            expected_saved = mean_cate * k
            gross = expected_saved * churn_revenue
            cost = k * contact_cost
            net = gross - cost
            roi = (net / cost * 100) if cost > 0 else 0.0

            rows.append(
                {
                    "Model": name,
                    "Target %": f"{int(pct * 100)}%",
                    "Customers Targeted": k,
                    "Avg Predicted CATE": round(mean_cate, 4),
                    "Expected Churners Saved": round(expected_saved, 1),
                    "Gross Revenue Saved ($)": round(gross, 0),
                    "Contact Cost ($)": round(cost, 0),
                    "Net Lift ($)": round(net, 0),
                    "ROI (%)": round(roi, 1),
                }
            )

    table = pd.DataFrame(rows)

    print("\n=== Business ROI Table ===")
    print(table.to_string(index=False))

    _ensure_dir(FIGURES_DIR.parent)
    csv_path = FIGURES_DIR.parent / "roi_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nROI table saved to {csv_path}")

    if drive_dir:
        _ensure_dir(drive_dir)
        table.to_csv(drive_dir / "roi_table.csv", index=False)

    _plot_roi_chart(table, drive_dir)
    return table


def _plot_roi_chart(table: pd.DataFrame, drive_dir: pathlib.Path | None):
    models = table["Model"].unique()
    percentiles = table["Target %"].unique()
    x = np.arange(len(percentiles))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        sub = table[table["Model"] == model]
        ax.bar(x + i * width, sub["Net Lift ($)"], width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(percentiles)
    ax.set_xlabel("Customer Segment Targeted")
    ax.set_ylabel("Net Lift ($)")
    ax.set_title(
        f"Expected Net Lift by Model and Target Segment\n"
        f"(Contact cost=${CONTACT_COST}/customer, Churn value=${CHURN_REVENUE}/customer)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "roi_net_lift.png", drive_dir)


# ---------------------------------------------------------------------------
# Fairness disparity report
# ---------------------------------------------------------------------------

def fairness_report(
    X_test: pd.DataFrame,
    cate_preds: dict,
    drive_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """
    Compute mean CATE by gender and senior citizen subgroup for each model.

    Encoding assumptions (LabelEncoder, alphabetical order):
      gender:        0 = Female, 1 = Male
      SeniorCitizen: 0 = Non-Senior (<65), 1 = Senior (≥65)

    A large disparity ratio (max_mean / min_mean) across subgroups may indicate
    that the model systematically underserves one demographic.
    """
    groups = {
        "gender": {0: "Female", 1: "Male"},
        "SeniorCitizen": {0: "Non-Senior", 1: "Senior"},
    }

    rows = []
    for model_name, cate in cate_preds.items():
        cate_arr = np.asarray(cate)
        for col, labels in groups.items():
            if col not in X_test.columns:
                continue
            subgroup_means = {}
            for val, label in labels.items():
                mask = (X_test[col].values == val)
                if mask.sum() == 0:
                    continue
                mean_cate = float(np.mean(cate_arr[mask]))
                n = int(mask.sum())
                rows.append(
                    {
                        "Model": model_name,
                        "Group": col,
                        "Subgroup": label,
                        "N": n,
                        "Mean CATE": round(mean_cate, 4),
                    }
                )
                subgroup_means[label] = mean_cate

            # Disparity ratio (max / min absolute mean)
            if len(subgroup_means) == 2:
                vals = list(subgroup_means.values())
                ratio = max(vals) / max(min(vals), 1e-9)
                rows[-1]["Disparity Ratio"] = round(ratio, 3)
                rows[-2]["Disparity Ratio"] = round(ratio, 3)

    table = pd.DataFrame(rows)

    if not table.empty:
        print("\n=== Fairness Disparity Report ===")
        print(table.to_string(index=False))

        _ensure_dir(FIGURES_DIR.parent)
        csv_path = FIGURES_DIR.parent / "fairness_report.csv"
        table.to_csv(csv_path, index=False)
        print(f"\nFairness report saved to {csv_path}")

        if drive_dir:
            _ensure_dir(drive_dir)
            table.to_csv(drive_dir / "fairness_report.csv", index=False)

    return table


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_evaluation(
    models,
    cate_preds: dict,
    auuc_scores: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    w_test: pd.Series,
    drive_dir: pathlib.Path | None = None,
):
    """Full evaluation pipeline called from the notebook."""
    print("=== Qini Curves ===")
    scores = plot_qini_curves(cate_preds, y_test, w_test, drive_dir)

    best_name = max(scores, key=scores.get)
    print(f"\nBest model by AUUC: {best_name} ({scores[best_name]:.4f})")

    best_model = next(m for m in models if m.name == best_name)

    print("\n=== SHAP Explanation (best model) ===")
    explain_best_model(best_model, X_train, X_test, drive_dir)

    print("\n=== ROI Analysis ===")
    table = roi_table(cate_preds, y_test, w_test, drive_dir=drive_dir)

    return scores, table
