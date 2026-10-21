"""Compare churn models, evaluate the winner once on a hold-out set, save artefacts.

Run from the project root:  python -m src.train
"""

import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split

from src.common import (
    DB_PATH,
    ID_COL,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TARGET,
    FEATURES,
    ensure_dirs,
    load_customers,
    make_pipeline,
)

TEST_SIZE = 0.2
CV_FOLDS = 5


def candidate_models() -> dict:
    """Fresh (unfitted) estimators. Class weights correct the 27/73 imbalance."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """All test-set metrics are computed here, once, for the selected model."""
    risk = model.predict_proba(X_test)[:, 1]
    predicted = (risk >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    return {
        "roc_auc": round(float(roc_auc_score(y_test, risk)), 4),
        "pr_auc": round(float(average_precision_score(y_test, risk)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "threshold": 0.5,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def gains_data(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Cumulative share of churners reached when contacting the riskiest X% first."""
    order = np.argsort(-y_score)
    churners = y_true[order]
    n = len(y_true)
    cumulative_caught = np.cumsum(churners) / churners.sum()
    shares = np.arange(1, n + 1) / n

    # Sample a clean 0..100% grid for the CSV and the chart.
    grid = np.linspace(0.0, 1.0, 101)
    caught = np.interp(grid, np.concatenate([[0.0], shares]), np.concatenate([[0.0], cumulative_caught]))
    return pd.DataFrame(
        {
            "share_contacted": grid.round(4),
            "share_of_churners_caught": caught.round(4),
            "random_baseline": grid.round(4),
        }
    )


def main() -> None:
    ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_customers()
    X = df[FEATURES]
    y = df[TARGET]

    # Hold out 20%. From here on, y_test is untouched until the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train: {len(X_train)}  test: {len(X_test)}  train churn rate: {y_train.mean():.4f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- model selection: cross-validation on the TRAINING set only ----------
    cv_rows = []
    for name, estimator in candidate_models().items():
        scores = cross_val_score(
            make_pipeline(estimator), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": round(float(scores.mean()), 4),
                "cv_roc_auc_std": round(float(scores.std()), 4),
            }
        )
        print(f"{name:24s} CV ROC-AUC {scores.mean():.4f} +/- {scores.std():.4f}")

    comparison = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    comparison["rank"] = comparison.index + 1
    comparison["selected"] = comparison.index == 0
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_name = comparison.loc[0, "model"]
    print(f"\nselected by CV: {best_name}")

    # --- single test-set evaluation ------------------------------------------
    best_model = make_pipeline(candidate_models()[best_name])
    best_model.fit(X_train, y_train)
    test_metrics = evaluate_on_test(best_model, X_test, y_test)
    print("test metrics:", json.dumps(test_metrics, indent=2))

    # --- gains curve ----------------------------------------------------------
    test_risk = best_model.predict_proba(X_test)[:, 1]
    gains = gains_data(y_test.to_numpy(), test_risk)
    gains.to_csv(REPORTS_DIR / "gains_curve.csv", index=False)
    at = {f"at_{int(s * 100)}pct": float(gains.loc[gains.share_contacted == s, "share_of_churners_caught"].iloc[0]) for s in (0.1, 0.2, 0.3)}
    print("gains:", at)

    # --- permutation importance on the test set, original feature names ------
    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": perm.importances_mean.round(4),
                "importance_std": perm.importances_std.round(4),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    print("\ntop permutation importances (test ROC-AUC drop):")
    print(importance.head(8).to_string(index=False))

    # --- out-of-fold scores for every customer (dashboard source) ------------
    # Each customer is scored by a model that never saw their label.
    oof_model = make_pipeline(candidate_models()[best_name])
    oof_risk = cross_val_predict(
        oof_model, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    scores = pd.DataFrame(
        {ID_COL: df[ID_COL].to_numpy(), "churn_risk": np.round(oof_risk, 6)}
    )
    assert scores[ID_COL].is_unique
    with sqlite3.connect(DB_PATH) as conn:
        scores.to_sql("customer_scores", conn, if_exists="replace", index=False)

    # --- final model on all data, for future scoring -------------------------
    final_model = make_pipeline(candidate_models()[best_name])
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    metrics = {
        "dataset": {
            "customers": int(len(df)),
            "churn_rate_pct": round(100 * float(y.mean()), 2),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "selection": {
            "method": "5-fold stratified CV ROC-AUC on the training set",
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        },
        "candidates": cv_rows,
        "best_model": best_name,
        "test": test_metrics,
        "gains": at,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nsaved: {REPORTS_DIR / 'metrics.json'}")
    print(f"saved: {MODELS_DIR / MODEL_PATH.name}")
    print(f"scored all {len(scores)} customers out-of-fold into customer_scores")


if __name__ == "__main__":
    main()
