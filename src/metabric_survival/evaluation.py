"""Cross-validation, discrimination metrics and paired model comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.metrics import cumulative_dynamic_auc

from .models import TunedCoxnet, harrell_c


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def make_folds(strata: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> list[Fold]:
    """Repeated stratified K-fold on the 3-level vital status.

    The same folds are reused for every endpoint, feature set and model, which is
    what makes the fold-level comparisons paired.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = []
    for i, (tr, te) in enumerate(splitter.split(np.zeros(len(strata)), strata)):
        folds.append(Fold(repeat=i // n_splits, fold=i % n_splits, train=tr, test=te))
    return folds


def time_dependent_auc(y_train, y_test, risk, months: list[float]) -> dict[str, float]:
    out = {}
    for m in months:
        try:
            auc, _ = cumulative_dynamic_auc(y_train, y_test, risk, [m])
            out[f"auc_{int(m)}m"] = float(auc[0])
        except ValueError:      # time point outside the test fold's follow-up range
            out[f"auc_{int(m)}m"] = np.nan
    return out


def cross_validate(
    make: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[Fold],
    td_months: list[float],
    keep_coefficients: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on each outer training fold, score on its test fold.

    Returns per-fold scores and (optionally) the non-zero elastic-net coefficients
    of every fold, used for selection frequencies.
    """
    rows, coefs = [], []
    for f in folds:
        pipe = make().fit(X.iloc[f.train], y[f.train])
        risk = pipe.predict(X.iloc[f.test])
        row = {"repeat": f.repeat, "fold": f.fold, "cindex": harrell_c(y[f.test], risk)}
        row.update(time_dependent_auc(y[f.train], y[f.test], risk, td_months))
        model = pipe.named_steps["model"]
        if isinstance(model, TunedCoxnet):
            row["best_alpha"] = model.best_alpha_
            row["n_nonzero"] = int(np.count_nonzero(model.coef_))
            if keep_coefficients:
                names = pipe.named_steps["prep"].get_feature_names_out()
                nz = np.flatnonzero(model.coef_)
                coefs.extend({"repeat": f.repeat, "fold": f.fold, "feature": names[i],
                              "coef": float(model.coef_[i])} for i in nz)
        if hasattr(model, "n_rounds_"):
            row["n_rounds"] = model.n_rounds_
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(coefs)


def corrected_resampled_ttest(diffs: np.ndarray, test_fraction: float) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated K-fold CV.

    Fold scores share training data, so the naive t-test is overconfident; the
    variance is inflated by (1/J + n_test/n_train).
    """
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    ratio = test_fraction / (1 - test_fraction)
    se = np.sqrt((1 / j + ratio) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    half = stats.t.ppf(0.975, df=j - 1) * se
    return {
        "mean_delta": float(d.mean()), "ci_low": float(d.mean() - half), "ci_high": float(d.mean() + half),
        "t": float(t), "p_value": float(p), "folds_better": int((d > 0).sum()), "n_folds": j,
    }


def summarize(scores: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    metrics = [c for c in scores.columns if c == "cindex" or c.startswith("auc_")]
    agg = scores.groupby(by, sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_folds"] = scores.groupby(by, sort=False).size()
    return agg.reset_index()


def iter_grid(cfg: dict[str, Any]) -> Iterator[tuple[str, str]]:
    for endpoint in cfg["endpoints"]:
        for feature_set in cfg["feature_sets"]:
            yield endpoint, feature_set
