import numpy as np
import pandas as pd

from metabric_survival.data import binarize_mutations, survival_target
from metabric_survival.preprocessing import build_preprocessor


def test_outcome_columns_never_in_any_feature_set(cfg, ds):
    forbidden = set(cfg["columns"]["outcomes"]) | set(cfg["columns"]["excluded"])
    for name, spec in cfg["feature_sets"].items():
        used = set(ds.columns.feature_set(spec["groups"]))
        assert not used & forbidden, f"{name} uses {used & forbidden}"


def test_preprocessor_output_has_no_outcome_columns(cfg, ds):
    groups = cfg["feature_sets"]["clinical_genomic"]["groups"]
    prep = build_preprocessor(groups, ds.columns, cfg).fit(ds.frame)
    names = " ".join(prep.get_feature_names_out())
    for col in cfg["columns"]["outcomes"]:
        assert col not in names


def test_removed_patients_are_recorded(ds):
    assert ds.n_raw == 1904
    assert ds.removed == {"missing outcome": 1, "zero follow-up": 1}
    assert len(ds.frame) == 1902
    assert (ds.frame["overall_survival_months"] > 0).all()


def test_gene_columns_match_file_layout(ds):
    assert len(ds.columns.expression) == 489
    assert len(ds.columns.mutations) == 173
    assert not set(ds.columns.expression) & set(ds.columns.mutations)


def test_mutation_binarization():
    df = pd.DataFrame({"tp53_mut": ["0", "R175H", "0", None, " 0 "]})
    out = binarize_mutations(df, ["tp53_mut"])
    assert out["tp53_mut"].tolist() == [0, 1, 0, 0, 0]


def test_endpoints_match_vital_status(cfg, ds):
    status = ds.frame["death_from_cancer"]
    assert survival_target(ds.frame, cfg, "dss")["event"].sum() == (status == "Died of Disease").sum()
    assert survival_target(ds.frame, cfg, "other_cause")["event"].sum() == (status == "Died of Other Causes").sum()
    os_ = survival_target(ds.frame, cfg, "os")
    assert os_["event"].sum() == (status != "Living").sum()
    # overall_survival == 1 exactly for living patients: the leak the audit documents
    assert np.array_equal(ds.frame["overall_survival"].to_numpy() == 1, (status == "Living").to_numpy())
