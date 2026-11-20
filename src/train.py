"""Train, tune and save the login anomaly model.
Covers: preprocessing (Exp 1), decision tree (Exp 3), GridSearchCV (Exp 4)."""
import json
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["hour", "failed_attempts", "new_device", "foreign_country",
            "vpn_used", "hours_since_last_login"]

def build_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # fill missing values
        ("scale", StandardScaler()),                    # normalize
        ("tree", DecisionTreeClassifier(random_state=42)),
    ])

def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

def main():
    df = pd.read_csv("data/logins.csv")
    X, y = df[FEATURES], df["is_anomaly"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Default model (baseline)
    base = build_pipeline().fit(X_tr, y_tr)
    base_m = evaluate(base, X_te, y_te)

    # Tuned model
    grid = GridSearchCV(
        build_pipeline(),
        {"tree__max_depth": [3, 5, 7, 10], "tree__min_samples_leaf": [1, 5, 10]},
        cv=5, scoring="f1")
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best_m = evaluate(best, X_te, y_te)

    # Simple fairness check: false-alarm rate for foreign vs local logins
    pred = best.predict(X_te)
    fair = {}
    for name, mask in [("foreign", X_te.foreign_country == 1),
                       ("local", X_te.foreign_country == 0)]:
        normal = mask & (y_te == 0)
        fair[name + "_false_alarm_rate"] = float((pred[normal] == 1).mean())

    joblib.dump(best, "models/model.joblib")
    results = {"best_params": grid.best_params_, "default_model": base_m,
               "tuned_model": best_m, "fairness_check": fair}
    json.dump(results, open("models/metrics.json", "w"), indent=2)

    print("Default  F1:", round(base_m["f1"], 3), " Acc:", round(base_m["accuracy"], 3))
    print("Tuned    F1:", round(best_m["f1"], 3), " Acc:", round(best_m["accuracy"], 3))
    print("Best params:", grid.best_params_)
    print("Fairness:", fair)

if __name__ == "__main__":
    main()
