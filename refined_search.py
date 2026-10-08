"""
Second-stage search for ML Assignment 1 (IMT2024077) - STAGE 2.

Reads outputs/top_configs_var<N>.csv produced by poly_regression.py, then:
  - takes the degrees in the top configs (plus neighbours)
  - runs a finer Ridge alpha search around the best Ridge alpha
  - runs Lasso (L1) on those degrees only, with a finer alpha grid
  - if Lasso wins, reports which polynomial terms / original features it kept
Uses the SAME hold-out split and CV folds (same SEED) as poly_regression.py, so CV MSE is
directly comparable. A *_refined.csv prediction file is written only if CV MSE improves.

Run poly_regression.py first, then:  python3 refine_search.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"  # silence Lasso ConvergenceWarning spam (also in workers)

import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ROLL = "IMT2024077"
DATA_DIR = "data"
OUT_DIR = "outputs"
SEED = 42          # must match poly_regression.py
K_FOLDS = 5        # must match poly_regression.py

PROBLEMS = {1: 10, 2: 20}   # max degree allowed per problem
TOP_K = 5                   # how many top configs to take degrees from
NEIGHBOURS = 1              # also try degrees +/- this far from those
RIDGE_MULTS = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10]  # x best Ridge alpha
LASSO_ALPHAS = [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3,
                1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 1e-1]


def analyse_lasso(best, cols):
    """Print which polynomial terms and original features survive in the Lasso model."""
    names = best.named_steps["poly"].get_feature_names_out(cols)
    coef = best.named_steps["model"].coef_
    nz = np.flatnonzero(coef)
    print(f"Lasso kept {len(nz)} of {coef.size} polynomial terms")

    order = nz[np.argsort(-np.abs(coef[nz]))][:10]
    print("strongest terms (coefficients on standardised terms):")
    for i in order:
        print(f"   {names[i]:<25s} {coef[i]:+.4f}")

    usage = {c: 0 for c in cols}
    for i in nz:
        for tok in names[i].split(" "):
            base = re.sub(r"\^\d+$", "", tok)
            if base in usage:
                usage[base] += 1
    print("number of kept terms involving each feature:", usage)


def run_problem(prb_id, max_degree):
    print(f"\n===== var{prb_id} refinement =====")
    top = pd.read_csv(f"{OUT_DIR}/top_configs_var{prb_id}.csv")
    top = top.sort_values("cv_mse").reset_index(drop=True)
    base_cv = top["cv_mse"].iloc[0]

    ridge_rows = top[top["model_name"] == "Ridge"]
    best_alpha = float(ridge_rows["param_model__alpha"].iloc[0]) if len(ridge_rows) else 1.0

    degs = set(int(d) for d in top.head(TOP_K)["param_poly__degree"])
    degs |= {d + k for d in list(degs) for k in range(-NEIGHBOURS, NEIGHBOURS + 1)}
    degrees = sorted(d for d in degs if 1 <= d <= max_degree)
    ridge_alphas = sorted({best_alpha * m for m in RIDGE_MULTS})
    print(f"stage-1 best CV MSE: {base_cv:.6f} | degrees to try: {degrees}")
    print(f"Ridge alphas: {[round(a, 4) for a in ridge_alphas]}")

    train = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_var{prb_id}.csv")
    test = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_var{prb_id}.csv")
    cols = [c for c in train.columns if c != "y"]
    X, y, X_test = train[cols].values, train["y"].values, test[cols].values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ])
    grid = [
        {"poly__degree": degrees, "model": [Ridge()], "model__alpha": ridge_alphas},
        {"poly__degree": degrees, "model": [Lasso(max_iter=100000)], "model__alpha": LASSO_ALPHAS},
    ]
    search = GridSearchCV(
        pipe, grid,
        cv=KFold(K_FOLDS, shuffle=True, random_state=SEED),
        scoring="neg_mean_squared_error", n_jobs=-1, error_score=np.nan,
    )
    search.fit(X_tr, y_tr)

    res = pd.DataFrame(search.cv_results_)
    res["model_name"] = res["param_model"].apply(lambda m: type(m).__name__)
    res["cv_mse"] = -res["mean_test_score"]
    res.sort_values("cv_mse").head(10)[
        ["param_poly__degree", "model_name", "param_model__alpha", "cv_mse"]
    ].to_csv(f"{OUT_DIR}/top_configs_refined_var{prb_id}.csv", index=False)

    p, best = search.best_params_, search.best_estimator_
    cv = -search.best_score_
    pred_val = best.predict(X_val)
    print("refined best -> degree:", p["poly__degree"],
          "| model:", type(p["model"]).__name__, "| alpha:", round(p["model__alpha"], 6))
    print(f"CV MSE: {cv:.6f} (stage 1: {base_cv:.6f})")
    print(f"hold-out MSE: {mean_squared_error(y_val, pred_val):.6f} | "
          f"hold-out R2: {r2_score(y_val, pred_val):.6f}")
    if isinstance(p["model"], Lasso):
        analyse_lasso(best, cols)

    if cv < base_cv:
        best.fit(X, y)
        out = f"{OUT_DIR}/{ROLL}_pred_var{prb_id}_refined.csv"
        pd.DataFrame({"y": best.predict(X_test)}).to_csv(out, index=False)
        print(f"improved on stage 1 -> saved {out}")
    else:
        print("no CV improvement over stage 1 -> keep your existing predictions")


if __name__ == "__main__":
    for prb_id, max_deg in PROBLEMS.items():
        run_problem(prb_id, max_deg)