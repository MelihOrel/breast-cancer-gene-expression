import numpy as np
from sksurv.util import Surv

from metabric_survival.evaluation import corrected_resampled_ttest, make_folds
from metabric_survival.models import TunedCoxnet, XGBCox, harrell_c


def _synthetic(n=400, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    hazard = np.exp(1.0 * X[:, 0] - 0.7 * X[:, 1])
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.exponential(1.5, size=n)
    y = Surv.from_arrays(event=event_time <= censor_time, time=np.minimum(event_time, censor_time) + 1e-3)
    return X, y


def test_tuned_coxnet_recovers_signal():
    X, y = _synthetic()
    model = TunedCoxnet(random_state=0).fit(X[:300], y[:300])
    assert model.coef_[0] > 0 and model.coef_[1] < 0
    assert harrell_c(y[300:], model.predict(X[300:])) > 0.7


def test_xgb_cox_beats_chance():
    X, y = _synthetic(seed=1)
    model = XGBCox(n_jobs=1, random_state=0).fit(X[:300], y[:300])
    assert harrell_c(y[300:], model.predict(X[300:])) > 0.65


def test_folds_are_disjoint_and_cover_everyone():
    strata = np.repeat([0, 1, 2], [50, 30, 20])
    folds = make_folds(strata, n_splits=5, n_repeats=2, seed=0)
    assert len(folds) == 10
    for r in range(2):
        tests = np.concatenate([f.test for f in folds if f.repeat == r])
        assert sorted(tests) == list(range(100))
    for f in folds:
        assert not set(f.train) & set(f.test)


def test_corrected_ttest_is_more_conservative_than_naive():
    d = np.random.default_rng(0).normal(0.01, 0.01, size=25)
    res = corrected_resampled_ttest(d, test_fraction=0.2)
    naive_se = d.std(ddof=1) / np.sqrt(len(d))
    assert (res["ci_high"] - res["ci_low"]) / 2 > 2 * naive_se
    assert res["n_folds"] == 25
