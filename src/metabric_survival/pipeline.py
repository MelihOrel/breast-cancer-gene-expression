"""Stage orchestration: each stage writes CSVs to results/ and PNGs to figures/."""
from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import plotting
from .audit import run_audit
from .data import Dataset, load_dataset, survival_target
from .descriptive import run_descriptive
from .evaluation import corrected_resampled_ttest, cross_validate, make_folds, summarize, time_dependent_auc
from .explain import coxnet_coefficients, shap_explanation
from .models import harrell_c, make_pipeline

STAGES = ("audit", "descriptive", "cv", "compare", "explain")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Runner:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.results = Path(cfg["paths"]["results"])
        self.figures = Path(cfg["paths"]["figures"])
        self.results.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.dpi = cfg["figures"]["dpi"]
        self.titles = cfg["figures"].get("titles", {}) or {}
        self.ds: Dataset = load_dataset(cfg)
        status = self.ds.frame[cfg["columns"]["status"]].to_numpy()
        c = cfg["cv"]
        self.folds = make_folds(status, c["n_splits"], c["n_repeats"], cfg["seed"])
        self.timings: dict[str, float] = {}
        _log(f"{len(self.ds.frame)} patients after removing {self.ds.removed} from {self.ds.n_raw}")

    def _csv(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.results / name, index=False, float_format="%.6g")

    # ------------------------------------------------------------------ stages
    def audit(self) -> None:
        summary, per_split, crosstab = run_audit(self.ds, self.cfg)
        self._csv(summary, "audit_summary.csv")
        self._csv(per_split, "audit_per_split.csv")
        self._csv(crosstab, "audit_status_vs_overall_survival.csv")
        plotting.plot_leakage_audit(summary, self.figures / "01_leakage_audit.png", self.dpi,
                                    title=self.titles.get("leakage_audit"))

    def descriptive(self) -> None:
        out = run_descriptive(self.ds.frame, self.cfg)
        self._csv(out["logrank"], "logrank_tests.csv")
        self._csv(out["km_summary"], "km_summary.csv")
        self._csv(out["at_risk_pam50"], "at_risk_pam50.csv")
        self._csv(out["at_risk_npi"], "at_risk_npi.csv")
        self._csv(out["cif_summary"], "competing_risks_cif.csv")
        self._csv(out["pam50_counts"], "pam50_counts.csv")
        (self.results / "cohort_facts.json").write_text(json.dumps(out["facts"], indent=2))
        lr = out["logrank"].set_index("grouping")

        def sub(g: str) -> str:
            p = lr.loc[g, "p_value"]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            return f"Kaplan-Meier; log-rank chi2 = {lr.loc[g, 'chi2']:.1f} on {lr.loc[g, 'df']} df, {ptxt}"

        max_m = self.cfg["descriptive"]["max_months"]
        plotting.plot_km(out["km_pam50"], out["at_risk_pam50"], plotting.CATEGORICAL,
                         self.titles.get("km_pam50", "Breast-cancer-specific survival by PAM50 subtype"),
                         sub("PAM50 subtype"), self.figures / "02_km_pam50.png", self.dpi, max_months=max_m)
        plotting.plot_km(out["km_npi"], out["at_risk_npi"], plotting.ORDINAL_3,
                         self.titles.get("km_npi", "Breast-cancer-specific survival by Nottingham Prognostic Index"),
                         sub("NPI group"), self.figures / "03_km_npi.png", self.dpi, max_months=max_m)
        plotting.plot_cif(out["cif"], self.figures / "04_competing_risks_age.png", self.dpi, max_months=max_m,
                          title=self.titles.get("competing_risks"))

    def cv(self) -> None:
        cfg, frame = self.cfg, self.ds.frame
        scores, coefs = [], []
        for endpoint in cfg["endpoints"]:
            y = survival_target(frame, cfg, endpoint)
            for fs in cfg["feature_sets"]:
                t0 = time.time()
                s, c = cross_validate(lambda fs=fs: make_pipeline("coxnet", fs, self.ds.columns, cfg),
                                      frame, y, self.folds, cfg["td_auc_months"], keep_coefficients=True)
                s.insert(0, "model", "coxnet")
                s.insert(0, "feature_set", fs)
                s.insert(0, "endpoint", endpoint)
                scores.append(s)
                if len(c):
                    c.insert(0, "feature_set", fs)
                    c.insert(0, "endpoint", endpoint)
                    coefs.append(c)
                _log(f"cv {endpoint:12s} {fs:18s} C={s['cindex'].mean():.3f} ({time.time() - t0:.0f}s)")
            scores.append(self._age_only(y, endpoint))
        scores = pd.concat(scores, ignore_index=True)
        self._csv(self._genes_vs_age(), "genes_only_risk_vs_age.csv")
        self._csv(scores, "cv_fold_scores.csv")
        self._csv(pd.concat(coefs, ignore_index=True), "coxnet_fold_coefficients.csv")
        summary = summarize(scores, ["endpoint", "feature_set", "model"])
        self._csv(summary, "cv_summary.csv")

        deltas, tests = self._paired(scores)
        self._csv(deltas, "paired_deltas.csv")
        self._csv(tests, "paired_tests.csv")
        plotting.plot_cindex_grid(summary, cfg, self.figures / "05_cindex_feature_sets.png", self.dpi,
                                  title=self.titles.get("cindex_grid"))
        plotting.plot_paired_deltas(deltas, tests, cfg, self.figures / "06_paired_deltas.png", self.dpi,
                                    title=self.titles.get("paired_deltas"))

    def _age_only(self, y: np.ndarray, endpoint: str) -> pd.DataFrame:
        """Reference row: age at diagnosis used directly as the risk score (nothing is fitted)."""
        age = self.ds.frame["age_at_diagnosis"].to_numpy(dtype=float)
        rows = []
        for f in self.folds:
            row = {"repeat": f.repeat, "fold": f.fold, "cindex": harrell_c(y[f.test], age[f.test])}
            row.update(time_dependent_auc(y[f.train], y[f.test], age[f.test], self.cfg["td_auc_months"]))
            rows.append(row)
        s = pd.DataFrame(rows)
        s.insert(0, "model", "none")
        s.insert(0, "feature_set", "age_only")
        s.insert(0, "endpoint", endpoint)
        return s

    def _genes_vs_age(self) -> pd.DataFrame:
        """How much of a genes-only risk score is just age? Out-of-fold scores, first CV repeat."""
        cfg, frame = self.cfg, self.ds.frame
        age = frame["age_at_diagnosis"].to_numpy(dtype=float)
        rows = []
        for endpoint in cfg["endpoints"]:
            y = survival_target(frame, cfg, endpoint)
            risk = np.full(len(frame), np.nan)
            for f in (f for f in self.folds if f.repeat == 0):
                pipe = make_pipeline("coxnet", "genomic", self.ds.columns, cfg).fit(frame.iloc[f.train], y[f.train])
                risk[f.test] = pipe.predict(frame.iloc[f.test])
            rho, p = spearmanr(risk, age)
            rows.append({"endpoint": endpoint, "spearman_rho_with_age": rho, "p_value": p,
                         "cindex_genes_only_oof": harrell_c(y, risk), "cindex_age_alone": harrell_c(y, age)})
        return pd.DataFrame(rows)

    def _paired(self, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        wide = scores.pivot_table(index=["endpoint", "repeat", "fold"], columns="feature_set", values="cindex")
        test_fraction = 1 / self.cfg["cv"]["n_splits"]
        deltas, tests = [], []
        for endpoint in self.cfg["endpoints"]:
            w = wide.loc[endpoint]
            for better, ref in self.cfg["paired_comparisons"]:
                d = (w[better] - w[ref]).to_numpy()
                deltas.append(pd.DataFrame({"endpoint": endpoint, "better": better, "reference": ref,
                                            "repeat": w.index.get_level_values("repeat"),
                                            "fold": w.index.get_level_values("fold"), "delta": d}))
                tests.append({"endpoint": endpoint, "better": better, "reference": ref,
                              **corrected_resampled_ttest(d, test_fraction)})
        return pd.concat(deltas, ignore_index=True), pd.DataFrame(tests)

    def compare(self) -> None:
        cfg, frame, mc = self.cfg, self.ds.frame, self.cfg["model_comparison"]
        y = survival_target(frame, cfg, mc["endpoint"])
        folds = [f for f in self.folds if f.repeat < mc["n_repeats"]]
        rows = []
        for model in mc["models"]:
            for fs in mc["feature_sets"]:
                t0 = time.time()
                s, _ = cross_validate(lambda m=model, fs=fs: make_pipeline(m, fs, self.ds.columns, cfg),
                                      frame, y, folds, cfg["td_auc_months"])
                s.insert(0, "model", model)
                s.insert(0, "feature_set", fs)
                s.insert(0, "endpoint", mc["endpoint"])
                rows.append(s)
                _log(f"compare {model:8s} {fs:18s} C={s['cindex'].mean():.3f} ({time.time() - t0:.0f}s)")
        scores = pd.concat(rows, ignore_index=True)
        self._csv(scores, "model_comparison_folds.csv")
        summary = summarize(scores, ["endpoint", "feature_set", "model"])
        self._csv(summary, "model_comparison_summary.csv")
        plotting.plot_model_comparison(summary, cfg, self.figures / "07_model_comparison.png", self.dpi,
                                       title=self.titles.get("model_comparison"))

    def explain(self) -> None:
        cfg, e = self.cfg, self.cfg["explain"]
        shap_out = shap_explanation(self.ds, cfg)
        self._csv(shap_out["features"], "shap_feature_importance.csv")
        self._csv(shap_out["groups"], "shap_group_share.csv")
        plotting.plot_shap_beeswarm(shap_out["explanation"], e["top_n"], self.figures / "08_shap_beeswarm.png", self.dpi)
        plotting.plot_shap_groups(shap_out["groups"], self.figures / "09_shap_group_share.png", self.dpi,
                                  title=self.titles.get("shap_groups"))

        fold_coefs = None
        path = self.results / "coxnet_fold_coefficients.csv"
        if path.exists():
            fc = pd.read_csv(path)
            fold_coefs = fc[(fc["endpoint"] == e["endpoint"]) & (fc["feature_set"] == e["feature_set"])]
        coefs, alpha = coxnet_coefficients(self.ds, cfg, fold_coefs)
        self._csv(coefs, "coxnet_coefficients_full_data.csv")
        plotting.plot_coefficients(coefs, 20, self.figures / "10_coxnet_coefficients.png", self.dpi)
        self.extra = {"xgb_cox_rounds_full_data": shap_out["n_rounds"], "coxnet_alpha_full_data": alpha,
                      "coxnet_nonzero_full_data": int(len(coefs))}

    # ------------------------------------------------------------------ driver
    def run(self, stages: list[str]) -> None:
        self.extra: dict[str, Any] = {}
        for stage in stages:
            _log(f"stage {stage} ...")
            t0 = time.time()
            getattr(self, stage)()
            self.timings[stage] = round(time.time() - t0, 1)
            _log(f"stage {stage} done in {self.timings[stage]}s")
        self._metadata(stages)

    def _metadata(self, stages: list[str]) -> None:
        cols = self.ds.columns
        meta_path = self.results / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update({
            "patients_raw": self.ds.n_raw, "patients_analysed": len(self.ds.frame),
            "removed": self.ds.removed,
            "n_features": {"clinical_numeric": len(cols.clinical_numeric),
                           "clinical_categorical": len(cols.clinical_categorical),
                           "molecular_subtypes": len(cols.molecular_subtypes),
                           "expression": len(cols.expression), "mutations": len(cols.mutations)},
            "events": {ep: int(survival_target(self.ds.frame, self.cfg, ep)["event"].sum())
                       for ep in self.cfg["endpoints"]},
            "seed": self.cfg["seed"], "quick": self.cfg["quick"],
            "cv": self.cfg["cv"], "n_outer_folds": len(self.folds),
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scikit-survival",
                                                 "xgboost", "shap", "matplotlib", "scipy")},
        })
        meta.setdefault("stage_seconds", {}).update(self.timings)
        meta.update(getattr(self, "extra", {}))
        meta_path.write_text(json.dumps(meta, indent=2))


def run(cfg: dict[str, Any], stages: list[str] | None = None) -> None:
    np.random.seed(cfg["seed"])
    plotting.apply_style()
    Runner(cfg).run(list(stages or STAGES))
