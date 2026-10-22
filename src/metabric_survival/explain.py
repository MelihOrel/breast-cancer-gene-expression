"""Model explanations: TreeSHAP on the boosted Cox model, elastic-net coefficients.

Explanations are computed on models refit to all patients. They describe what
the models rely on; the performance numbers come from the cross-validation only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from .data import Dataset, survival_target
from .models import make_pipeline
from .preprocessing import feature_group, pretty_name


def shap_explanation(ds: Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
    e = cfg["explain"]
    y = survival_target(ds.frame, cfg, e["endpoint"])
    pipe = make_pipeline("xgb_cox", e["feature_set"], ds.columns, cfg).fit(ds.frame, y)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = prep.transform(ds.frame)
    names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(model.booster_)
    values = explainer.shap_values(Z)              # log-hazard contributions
    mean_abs = np.abs(values).mean(axis=0)

    features = pd.DataFrame({
        "feature": names,
        "label": [pretty_name(n, ds.columns) for n in names],
        "group": [feature_group(n) for n in names],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    groups = features.groupby("group")["mean_abs_shap"].agg(["sum", "size"]).rename(
        columns={"sum": "total_mean_abs_shap", "size": "n_features"})
    groups["share"] = groups["total_mean_abs_shap"] / groups["total_mean_abs_shap"].sum()
    groups = groups.sort_values("share", ascending=False).reset_index()

    order = np.argsort(-mean_abs)
    explanation = shap.Explanation(values=values, base_values=np.zeros(len(Z)), data=Z,
                                   feature_names=[pretty_name(n, ds.columns) for n in names])
    return {"features": features, "groups": groups, "explanation": explanation,
            "order": order, "n_rounds": model.n_rounds_}


def coxnet_coefficients(ds: Dataset, cfg: dict[str, Any],
                        fold_coefs: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    """Coefficients of the elastic-net Cox model refit on all patients (and its penalty),
    plus how often each feature was selected (non-zero) across the outer CV folds."""
    e = cfg["explain"]
    y = survival_target(ds.frame, cfg, e["endpoint"])
    pipe = make_pipeline("coxnet", e["feature_set"], ds.columns, cfg).fit(ds.frame, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    out = pd.DataFrame({
        "feature": names,
        "label": [pretty_name(n, ds.columns) for n in names],
        "group": [feature_group(n) for n in names],
        "coef": model.coef_,
        "hazard_ratio_per_sd": np.exp(model.coef_),
    })
    if fold_coefs is not None and len(fold_coefs):
        n_folds = fold_coefs[["repeat", "fold"]].drop_duplicates().shape[0]
        freq = fold_coefs.groupby("feature").size() / n_folds
        out["selection_frequency"] = out["feature"].map(freq).fillna(0.0)
    out = out[out["coef"] != 0].copy()
    out["abs_coef"] = out["coef"].abs()
    out = out.sort_values("abs_coef", ascending=False, ignore_index=True).drop(columns="abs_coef")
    return out, float(model.best_alpha_)
