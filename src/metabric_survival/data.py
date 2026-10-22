"""Loading, cleaning and outcome construction for the METABRIC file.

Two patients are removed and both removals are recorded:
  * one has no value for ``death_from_cancer`` (the outcome is unknown, so it is
    not imputed; the original notebook filled it with the mode);
  * one has 0 months of follow-up, which contributes nothing to any survival
    likelihood and is not a valid label for XGBoost's Cox objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sksurv.util import Surv

LIVING = "Living"
DISEASE = "Died of Disease"
OTHER = "Died of Other Causes"
STATUS_LEVELS = (LIVING, DISEASE, OTHER)


@dataclass
class FeatureColumns:
    """Column names of each feature group, taken from the config and the file layout."""

    clinical_numeric: list[str]
    clinical_categorical: list[str]
    molecular_subtypes: list[str]
    expression: list[str]
    mutations: list[str]
    mutation_numeric: list[str]
    display_names: dict[str, str] = field(default_factory=dict)

    def label(self, col: str) -> str:
        return self.display_names.get(col, col)

    def group(self, name: str) -> list[str]:
        groups = {
            "clinical": self.clinical_numeric + self.clinical_categorical,
            "subtypes": self.molecular_subtypes,
            "expression": self.expression,
            "mutations": self.mutations + self.mutation_numeric,
        }
        return groups[name]

    @property
    def categorical(self) -> list[str]:
        return self.clinical_categorical + self.molecular_subtypes

    def feature_set(self, groups: list[str]) -> list[str]:
        cols: list[str] = []
        for g in groups:
            cols.extend(self.group(g))
        return cols


@dataclass
class Dataset:
    frame: pd.DataFrame
    columns: FeatureColumns
    removed: dict[str, int] = field(default_factory=dict)
    n_raw: int = 0


def read_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["data"], low_memory=False)


def gene_columns(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    c = cfg["columns"]
    genes = list(raw.columns[c["first_gene_column"]:])
    mutations = [g for g in genes if g.endswith(c["mutation_suffix"])]
    expression = [g for g in genes if not g.endswith(c["mutation_suffix"])]
    return expression, mutations


def binarize_mutations(frame: pd.DataFrame, mutations: list[str]) -> pd.DataFrame:
    """Raw mutation cells hold '0' (wild type) or a protein change such as 'R175H'."""
    out = frame.copy()
    for col in mutations:
        out[col] = (out[col].fillna("0").astype(str).str.strip() != "0").astype(np.int8)
    return out


def load_dataset(cfg: dict[str, Any]) -> Dataset:
    raw = read_raw(cfg)
    c = cfg["columns"]
    expression, mutations = gene_columns(raw, cfg)

    removed = {}
    keep = raw[c["status"]].notna()
    removed["missing outcome"] = int((~keep).sum())
    frame = raw[keep]
    positive_time = frame[c["time"]] > 0
    removed["zero follow-up"] = int((~positive_time).sum())
    frame = frame[positive_time].reset_index(drop=True)

    frame = binarize_mutations(frame, mutations)
    categorical = c["clinical_categorical"] + c["molecular_subtypes"]
    # Plain object dtype keeps the behaviour identical under pandas 2 and 3.
    for col in categorical:
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)

    columns = FeatureColumns(
        clinical_numeric=list(c["clinical_numeric"]),
        clinical_categorical=list(c["clinical_categorical"]),
        molecular_subtypes=list(c["molecular_subtypes"]),
        expression=expression,
        mutations=mutations,
        mutation_numeric=list(c["mutation_numeric"]),
        display_names=dict(c.get("display_names") or {}),
    )
    return Dataset(frame=frame, columns=columns, removed=removed, n_raw=len(raw))


def survival_target(frame: pd.DataFrame, cfg: dict[str, Any], endpoint: str) -> np.ndarray:
    c = cfg["columns"]
    events = cfg["endpoints"][endpoint]["event"]
    return Surv.from_arrays(
        event=frame[c["status"]].isin(events).to_numpy(),
        time=frame[c["time"]].to_numpy(dtype=float),
    )


def competing_event_codes(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """0 = alive at last follow-up, 1 = died of breast cancer, 2 = died of another cause."""
    status = frame[cfg["columns"]["status"]]
    return status.map({LIVING: 0, DISEASE: 1, OTHER: 2}).to_numpy(dtype=int)
