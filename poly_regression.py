"""
Polynomial regression for ML Assignment 1 (IMT2024077) - STAGE 1.

For each problem (var1, var2):
  1. Load train/test CSV.
  2. Hold out 20% of train as a sanity-check (validation) set.
  3. K-fold CV on the other 80% chooses degree, model type (least squares / Ridge / Lasso)
     and alpha.
  4. Degree sweep: for EVERY degree 1..max, fit plain least squares and Ridge (best alpha
     for that degree from CV) and record train MSE, validation MSE and validation R2.
     -> outputs/degree_sweep_var<N>.csv and .png  (use these for the report)
  5. Refit the best config on the FULL train set and write <ROLL>_pred_var<N>.csv.

Run:  python3 poly_regression.py
Layout:  data/IMT2024077_train_var1.csv, data/IMT2024077_test_var1.csv, ... (same for var2)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ROLL = "IMT2024077"
DATA_DIR = "data"
OUT_DIR = "outputs"
SEED = 42
K_FOLDS = 5
USE_LASSO = False  # Lasso is handled in refine_search.py (much faster there)

PROBLEMS = {1: 10, 2: 20}  # max degree stated in the assignment

ALPHAS_RIDGE = [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]
ALPHAS_LASSO = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def make_pipe(degree, model):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def build_grid(max_degree):
    degrees = list(range(1, max_degree + 1))
    grid = [
        {"poly__degree": degrees, "model": [LinearRegression()]},
        {"poly__degree": degrees, "model": [Ridge()], "model__alpha": ALPHAS_RIDGE},
    ]
    if USE_LASSO:
        grid.append({"poly__degree": degrees,
                     "model": [Lasso(max_iter=50000)],
                     "model__alpha": ALPHAS_LASSO})
    return grid


def degree_sweep(res, X_tr, X_val, y_tr, y_val, max_degree):
    """Per-degree train/validation metrics for plain least squares and Ridge."""
    ridge_res = res[res["model_name"] == "Ridge"]
    rows = []
    for d in range(1, max_degree + 1):
        ols = make_pipe(d, LinearRegression()).fit(X_tr, y_tr)
        p_tr, p_val = ols.predict(X_tr), ols.predict(X_val)
        row = {
            "degree": d,
            "n_terms": ols.named_steps["poly"].n_output_features_,
            "ols_train_mse": mean_squared_error(y_tr, p_tr),
            "ols_val_mse": mean_squared_error(y_val, p_val),
            "ols_val_r2": r2_score(y_val, p_val),
        }
        best = ridge_res[ridge_res["param_poly__degree"] == d].sort_values("cv_mse").iloc[0]
        alpha = float(best["param_model__alpha"])
        rdg = make_pipe(d, Ridge(alpha=alpha)).fit(X_tr, y_tr)
        p_tr, p_val = rdg.predict(X_tr), rdg.predict(X_val)
        row.update({
            "ridge_alpha": alpha,
            "ridge_cv_mse": best["cv_mse"],
            "ridge_train_mse": mean_squared_error(y_tr, p_tr),
            "ridge_val_mse": mean_squared_error(y_val, p_val),
            "ridge_val_r2": r2_score(y_val, p_val),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_problem(prb_id, max_degree):
    print(f"\n===== var{prb_id} (max degree {max_degree}) =====")
    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_var{prb_id}.csv")
    test = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_var{prb_id}.csv")

    feature_cols = [c for c in train.columns if c != "y"]
    X, y = train[feature_cols].values, train["y"].values
    X_test = test[feature_cols].values
    print(f"train: {X.shape}, test: {X_test.shape}, features: {feature_cols}")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    search = GridSearchCV(
        make_pipe(1, LinearRegression()), build_grid(max_degree),
        cv=KFold(K_FOLDS, shuffle=True, random_state=SEED),
        scoring="neg_mean_squared_error",
        n_jobs=-1, error_score=np.nan,
    )
    search.fit(X_tr, y_tr)

    best = search.best_estimator_
    p = search.best_params_
    print("best degree:", p["poly__degree"],
          "| model:", type(p["model"]).__name__,
          "| alpha:", p.get("model__alpha", "-"))
    print(f"CV MSE (K={K_FOLDS}): {-search.best_score_:.6f}")
    pred_val = best.predict(X_val)
    print(f"hold-out MSE: {mean_squared_error(y_val, pred_val):.6f} | "
          f"hold-out R2: {r2_score(y_val, pred_val):.6f}")

    res = pd.DataFrame(search.cv_results_)
    res["model_name"] = res["param_model"].apply(lambda m: type(m).__name__)
    res["cv_mse"] = -res["mean_test_score"]
    res["param_poly__degree"] = res["param_poly__degree"].astype(int)
    os.makedirs(OUT_DIR, exist_ok=True)

    # top configs (read by refine_search.py)
    res.sort_values("cv_mse").head(10)[
        ["param_poly__degree", "model_name", "param_model__alpha", "cv_mse"]
    ].to_csv(f"{OUT_DIR}/top_configs_var{prb_id}.csv", index=False)

    # CV error vs degree
    plt.figure(figsize=(7, 4))
    for name, g in res.groupby("model_name"):
        curve = g.groupby("param_poly__degree")["cv_mse"].min()
        plt.plot(curve.index, curve.values, marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("CV MSE (log scale)")
    plt.title(f"var{prb_id}: CV error vs degree")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cv_vs_degree_var{prb_id}.png", dpi=150)
    plt.close()

    # degree-by-degree table for the report
    sweep = degree_sweep(res, X_tr, X_val, y_tr, y_val, max_degree)
    sweep.to_csv(f"{OUT_DIR}/degree_sweep_var{prb_id}.csv", index=False)
    print("\nDegree sweep (validation = 20% hold-out):")
    print(sweep[["degree", "n_terms", "ols_train_mse", "ols_val_mse", "ols_val_r2",
                 "ridge_alpha", "ridge_val_mse", "ridge_val_r2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plt.figure(figsize=(7, 4))
    plt.plot(sweep["degree"], sweep["ols_train_mse"], "k--", marker=".", label="Least squares: train")
    plt.plot(sweep["degree"], sweep["ols_val_mse"], marker="o", label="Least squares: validation")
    plt.plot(sweep["degree"], sweep["ridge_val_mse"], marker="s", label="Ridge: validation")
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE (log scale)")
    plt.title(f"var{prb_id}: bias-variance across degrees")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/degree_sweep_var{prb_id}.png", dpi=150)
    plt.close()

    # refit chosen config on the full training set and predict the test set
    best.fit(X, y)
    preds = best.predict(X_test)
    pd.DataFrame({"y": preds}).to_csv(f"{OUT_DIR}/{ROLL}_pred_var{prb_id}.csv", index=False)
    print(f"\nsaved {OUT_DIR}/{ROLL}_pred_var{prb_id}.csv ({len(preds)} rows)")


if __name__ == "__main__":
    for prb_id, max_deg in PROBLEMS.items():
        run_problem(prb_id, max_deg)