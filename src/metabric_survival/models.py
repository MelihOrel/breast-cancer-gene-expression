"""Survival models with sklearn-style fit/predict, usable inside a Pipeline.

* ``TunedCoxnet``  elastic-net Cox; the penalty is chosen by an inner, stratified
  K-fold CV over a warm-started alpha path computed on the training data only.
* ``XGBCox``       gradient-boosted Cox model; the number of trees is chosen by early
  stopping on a held-out slice of the training data, then refit on all of it.
* Random survival forest comes straight from scikit-survival.

``predict`` always returns a risk score: higher means an earlier expected event.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from .data import FeatureColumns
from .preprocessing import build_preprocessor


def harrell_c(y: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


class TunedCoxnet(BaseEstimator):
    def __init__(self, l1_ratio=0.5, n_alphas=25, alpha_min_ratio=0.02, max_iter=20000,
                 inner_splits=3, random_state=42):
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.max_iter = max_iter
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _coxnet(self, alphas=None) -> CoxnetSurvivalAnalysis:
        kw = dict(l1_ratio=self.l1_ratio, max_iter=self.max_iter, tol=1e-6)
        if alphas is None:
            return CoxnetSurvivalAnalysis(n_alphas=self.n_alphas, alpha_min_ratio=self.alpha_min_ratio, **kw)
        return CoxnetSurvivalAnalysis(alphas=alphas, **kw)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        path = self._coxnet().fit(X, y)            # full-data path defines the alpha grid
        grid = path.alphas_
        score = np.zeros(len(grid))
        inner = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        for tr, va in inner.split(X, y["event"]):
            fold = self._coxnet(alphas=grid).fit(X[tr], y[tr])
            risks = X[va] @ fold.coef_                 # one column of risk scores per alpha
            score += [harrell_c(y[va], risks[:, k]) for k in range(len(grid))]
        best = int(np.argmax(score))
        self.alphas_ = grid
        self.inner_cindex_ = score / self.inner_splits
        self.best_alpha_ = float(grid[best])
        self.coef_ = path.coef_[:, best].copy()
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_


class XGBCox(BaseEstimator):
    def __init__(self, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5,
                 min_child_weight=5, reg_lambda=1.0, max_rounds=2000, early_stopping_rounds=50,
                 validation_fraction=0.2, n_jobs=-1, random_state=42):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _params(self) -> dict[str, Any]:
        return {
            "objective": "survival:cox", "eval_metric": "cox-nloglik", "tree_method": "hist",
            "eta": self.learning_rate, "max_depth": self.max_depth, "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree, "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda, "seed": self.random_state,
            "nthread": os.cpu_count() if self.n_jobs in (None, -1) else self.n_jobs,
        }

    @staticmethod
    def _labels(y) -> np.ndarray:
        # XGBoost convention: positive time = event, negative time = right-censored.
        return np.where(y["event"], y["time"], -y["time"]).astype(float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        label = self._labels(y)
        tr, va = train_test_split(np.arange(len(X)), test_size=self.validation_fraction,
                                  stratify=y["event"], random_state=self.random_state)
        probe = xgb.train(self._params(), xgb.DMatrix(X[tr], label=label[tr]),
                          num_boost_round=self.max_rounds,
                          evals=[(xgb.DMatrix(X[va], label=label[va]), "val")],
                          early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
        self.n_rounds_ = int(probe.best_iteration) + 1
        self.booster_ = xgb.train(self._params(), xgb.DMatrix(X, label=label),
                                  num_boost_round=self.n_rounds_)
        return self

    def predict(self, X):
        return self.booster_.predict(xgb.DMatrix(np.asarray(X, dtype=float)), output_margin=True)


def make_model(name: str, cfg: dict[str, Any]) -> BaseEstimator:
    seed, n_jobs = cfg["seed"], cfg["n_jobs"]
    params = dict(cfg["models"][name])
    if name == "coxnet":
        return TunedCoxnet(inner_splits=cfg["cv"]["inner_splits"], random_state=seed, **params)
    if name == "rsf":
        return RandomSurvivalForest(n_jobs=n_jobs, random_state=seed, low_memory=True, **params)
    if name == "xgb_cox":
        return XGBCox(n_jobs=n_jobs, random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(model: str, feature_set: str, cols: FeatureColumns, cfg: dict[str, Any]) -> Pipeline:
    groups = cfg["feature_sets"][feature_set]["groups"]
    return Pipeline([
        ("prep", build_preprocessor(groups, cols, cfg)),
        ("model", make_model(model, cfg)),
    ])
