"""Audit of the original notebook's 3-class classifier.

The original target was ``death_from_cancer`` (Living / Died of Disease /
Died of Other Causes), yet ``overall_survival`` and ``overall_survival_months``
stayed among the features. ``overall_survival`` is 1 exactly for the "Living"
class, so the model could read part of the answer from its input.

Here the original model (multinomial logistic regression, C as selected by the
original grid search) is re-run with every preprocessing step inside the CV
folds, once with the leaking columns and twice with them removed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import STATUS_LEVELS, Dataset

VARIANTS = {
    "original_features": "As in the original notebook (includes overall_survival and overall_survival_months)",
    "without_overall_survival": "overall_survival removed",
    "without_survival_columns": "overall_survival and overall_survival_months removed (leak-free)",
    "majority_baseline": "Always predicts the most frequent class",
}


def _audit_columns(ds: Dataset, cfg: dict[str, Any], variant: str) -> tuple[list[str], list[str]]:
    """Feature columns the original notebook used: everything except the id and the target."""
    c = cfg["columns"]
    cols = ds.columns
    categorical = cols.categorical + ["cancer_type", "oncotree_code"]
    numeric = cols.clinical_numeric + ["cohort"] + cols.mutation_numeric + cols.expression + cols.mutations
    if variant == "original_features":
        numeric += ["overall_survival", "overall_survival_months"]
    elif variant == "without_overall_survival":
        numeric += ["overall_survival_months"]
    assert c["status"] not in numeric + categorical
    return numeric, categorical


def _pipeline(numeric: list[str], categorical: list[str], cfg: dict[str, Any]) -> Pipeline:
    a = cfg["audit"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    clf = LogisticRegression(C=a["logistic_C"], max_iter=a["max_iter"])
    return Pipeline([("prep", prep), ("clf", clf)])


def run_audit(ds: Dataset, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = cfg["audit"]
    frame = ds.frame
    y = frame[cfg["columns"]["status"]].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=a["n_splits"], n_repeats=a["n_repeats"], random_state=cfg["seed"])
    folds = list(cv.split(frame, y))

    rows = []
    for variant in VARIANTS:
        if variant == "majority_baseline":
            make = lambda: DummyClassifier(strategy="most_frequent")  # noqa: E731
        else:
            numeric, categorical = _audit_columns(ds, cfg, variant)
            make = lambda n=numeric, c=categorical: _pipeline(n, c, cfg)  # noqa: E731
        for i, (tr, te) in enumerate(folds):
            model = make().fit(frame.iloc[tr], y[tr])
            pred = model.predict(frame.iloc[te])
            recalls = recall_score(y[te], pred, labels=list(STATUS_LEVELS), average=None, zero_division=0)
            row = {"variant": variant, "split": i,
                   "accuracy": accuracy_score(y[te], pred),
                   "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0)}
            row.update({f"recall_{lvl}": r for lvl, r in zip(STATUS_LEVELS, recalls)})
            rows.append(row)

    per_split = pd.DataFrame(rows)
    metrics = [c for c in per_split.columns if c not in ("variant", "split")]
    summary = per_split.groupby("variant", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, "description", summary["variant"].map(VARIANTS))

    crosstab = pd.crosstab(frame[cfg["columns"]["status"]], frame["overall_survival"])
    crosstab.columns = [f"overall_survival={c}" for c in crosstab.columns]
    return summary, per_split, crosstab.reset_index()
