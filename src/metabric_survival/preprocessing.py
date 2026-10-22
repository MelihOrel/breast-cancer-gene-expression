"""Feature preprocessing, always fitted inside a training fold.

Each feature group gets its own named branch, so transformed column names carry
their group (``clin_num__``, ``clin_cat__``, ``subtype_cat__``, ``expr__``,
``mut__``, ``mut_num__``). That is what lets SHAP values and coefficients be
summed back to groups later.
"""
from __future__ import annotations

import re
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FeatureColumns

BRANCH_GROUP = {
    "clin_num": "Clinical",
    "clin_cat": "Clinical",
    "subtype_cat": "Molecular subtypes",
    "expr": "Gene expression",
    "mut": "Mutations",
    "mut_num": "Mutations",
}


def _numeric_with_missing() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])


def _categorical(min_frequency: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_frequency, sparse_output=False)),
        ("scale", StandardScaler()),
    ])


def build_preprocessor(groups: list[str], cols: FeatureColumns, cfg: dict[str, Any]) -> ColumnTransformer:
    p = cfg["preprocessing"]
    prevalence = p["min_mutation_prevalence"]
    branches = []
    if "clinical" in groups:
        branches.append(("clin_num", _numeric_with_missing(), cols.clinical_numeric))
        branches.append(("clin_cat", _categorical(p["min_category_frequency"]), cols.clinical_categorical))
    if "subtypes" in groups:
        branches.append(("subtype_cat", _categorical(p["min_category_frequency"]), cols.molecular_subtypes))
    if "expression" in groups:
        branches.append(("expr", StandardScaler(), cols.expression))
    if "mutations" in groups:
        # Bernoulli variance p(1-p): keeps flags whose training-fold prevalence exceeds ~1%.
        branches.append(("mut", Pipeline([
            ("prevalence", VarianceThreshold(threshold=prevalence * (1 - prevalence))),
            ("scale", StandardScaler()),
        ]), cols.mutations))
        branches.append(("mut_num", _numeric_with_missing(), cols.mutation_numeric))
    if not branches:
        raise ValueError(f"No feature groups selected: {groups}")
    return ColumnTransformer(branches, remainder="drop")


def feature_group(name: str) -> str:
    """Group label of a transformed column name such as ``expr__brca1``."""
    return BRANCH_GROUP[name.split("__", 1)[0]]


def pretty_name(name: str, cols: FeatureColumns) -> str:
    """Readable label for a transformed column name."""
    branch, rest = name.split("__", 1)
    if branch == "expr":
        return f"{rest.upper()} expression"
    if branch == "mut":
        return f"{rest[:-len('_mut')].upper()} mutation" if rest.endswith("_mut") else rest
    if rest.startswith("missingindicator_"):
        return f"{cols.label(rest[len('missingindicator_'):])} missing"
    if branch in ("clin_cat", "subtype_cat"):
        source = cols.clinical_categorical if branch == "clin_cat" else cols.molecular_subtypes
        for col in sorted(source, key=len, reverse=True):
            if rest.startswith(col + "_"):
                level = rest[len(col) + 1:].replace("infrequent_sklearn", "other (rare)")
                if re.fullmatch(r"[A-Z]{4,}( [A-Z]+)*", level):   # e.g. MASTECTOMY, BREAST CONSERVING
                    level = level.capitalize()
                return f"{cols.label(col)} = {level}"
    return cols.label(rest)
