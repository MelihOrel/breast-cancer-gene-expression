"""Kaplan-Meier curves, log-rank tests and competing-risk cumulative incidence."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import cumulative_incidence_competing_risks, kaplan_meier_estimator

from .data import competing_event_codes, survival_target


def npi_group(npi: pd.Series, cutoffs: list[float]) -> pd.Series:
    """Good < 3.4 <= Moderate <= 5.4 < Poor (Galea et al., 1992)."""
    lo, hi = cutoffs
    labels = np.select([npi < lo, npi > hi],
                       [f"Good (NPI < {lo})", f"Poor (NPI > {hi})"],
                       default=f"Moderate (NPI {lo}-{hi})")
    return pd.Series(labels, index=npi.index)


def age_group(age: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(age, bins=bins, labels=labels, right=False).astype(str)


def kaplan_meier_by_group(frame: pd.DataFrame, y: np.ndarray, groups: pd.Series, order: list[str],
                          at_risk_months: list[float]) -> tuple[dict[str, Any], pd.DataFrame]:
    """KM curve, median survival and numbers at risk for each group in ``order``."""
    curves, at_risk_rows = {}, []
    for g in order:
        mask = (groups == g).to_numpy()
        t, s, ci = kaplan_meier_estimator(y["event"][mask], y["time"][mask], conf_type="log-log")
        curves[g] = {"time": t, "surv": s, "ci_low": ci[0], "ci_high": ci[1], "n": int(mask.sum()),
                     "events": int(y["event"][mask].sum())}
        times = y["time"][mask]
        at_risk_rows.append({"group": g, **{f"{int(m)}m": int((times >= m).sum()) for m in at_risk_months}})
    return curves, pd.DataFrame(at_risk_rows)


def survival_at(curve: dict[str, Any], months: float) -> float:
    idx = np.searchsorted(curve["time"], months, side="right") - 1
    return float(curve["surv"][idx]) if idx >= 0 else 1.0


def logrank(y: np.ndarray, groups: pd.Series, order: list[str]) -> dict[str, float]:
    mask = groups.isin(order).to_numpy()
    chi2, p = compare_survival(y[mask], groups.to_numpy()[mask])
    return {"chi2": float(chi2), "df": len(order) - 1, "p_value": float(p), "n": int(mask.sum())}


def cohort_facts(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Follow-up, age and missing-data facts quoted in the README."""
    c = cfg["columns"]
    status, time = frame[c["status"]], frame[c["time"]]
    living = time[status == "Living"]
    lat_missing = frame["primary_tumor_laterality"].isna()
    disease = status == "Died of Disease"
    return {
        "max_followup_months": float(time.max()),
        "living_followup_months": {"min": float(living.min()), "median": float(living.median()),
                                   "max": float(living.max())},
        "living_followed_under_60_months": int((living < 60).sum()),
        "median_age_at_diagnosis_by_status": frame.groupby(c["status"])["age_at_diagnosis"].median().round(1).to_dict(),
        "laterality_missing": {
            "n": int(lat_missing.sum()),
            "by_cohort": {str(int(k)): int(v) for k, v in frame.loc[lat_missing, "cohort"].value_counts().sort_index().items()},
            "breast_cancer_death_rate_if_missing": float(disease[lat_missing].mean()),
            "breast_cancer_death_rate_if_recorded": float(disease[~lat_missing].mean()),
        },
    }


def run_descriptive(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    d = cfg["descriptive"]
    y = survival_target(frame, cfg, d["endpoint"])

    pam50 = frame[d["pam50_column"]].astype(str)
    km_pam50, risk_pam50 = kaplan_meier_by_group(frame, y, pam50, d["pam50_order"], d["at_risk_months"])

    npi = npi_group(frame["nottingham_prognostic_index"], d["npi_cutoffs"])
    npi_order = sorted(npi.unique(), key=lambda s: ["Good", "Moderate", "Poor"].index(s.split()[0]))
    km_npi, risk_npi = kaplan_meier_by_group(frame, y, npi, npi_order, d["at_risk_months"])

    tests = pd.DataFrame([
        {"grouping": "PAM50 subtype", **logrank(y, pam50, d["pam50_order"])},
        {"grouping": "NPI group", **logrank(y, npi, npi_order)},
    ])

    km_rows = []
    for name, curves in (("PAM50 subtype", km_pam50), ("NPI group", km_npi)):
        for g, c in curves.items():
            km_rows.append({"grouping": name, "group": g, "n": c["n"], "events": c["events"],
                            "dss_5y": survival_at(c, 60), "dss_10y": survival_at(c, 120),
                            "dss_15y": survival_at(c, 180)})
    pam50_counts = pam50.value_counts().rename_axis("subtype").reset_index(name="n")

    # Competing risks: cumulative incidence of breast-cancer death vs other-cause death.
    codes = competing_event_codes(frame, cfg)
    ages = age_group(frame["age_at_diagnosis"], d["age_bins"], d["age_labels"])
    cif, cif_rows = {}, []
    for g in d["age_labels"]:
        mask = (ages == g).to_numpy()
        t, inc = cumulative_incidence_competing_risks(codes[mask], frame[cfg["columns"]["time"]].to_numpy()[mask])
        cif[g] = {"time": t, "disease": inc[1], "other": inc[2], "n": int(mask.sum())}
        for m in d["cif_months"]:
            idx = np.searchsorted(t, m, side="right") - 1
            at = (lambda row: float(inc[row][idx]) if idx >= 0 else 0.0)
            cif_rows.append({"age_group": g, "n": int(mask.sum()), "months": m,
                             "cif_breast_cancer_death": at(1), "cif_other_cause_death": at(2)})

    return {
        "facts": cohort_facts(frame, cfg),
        "km_pam50": km_pam50, "at_risk_pam50": risk_pam50,
        "km_npi": km_npi, "at_risk_npi": risk_npi,
        "logrank": tests, "km_summary": pd.DataFrame(km_rows), "pam50_counts": pam50_counts,
        "cif": cif, "cif_summary": pd.DataFrame(cif_rows),
    }
