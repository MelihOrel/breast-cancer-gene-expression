import pytest

from metabric_survival.config import load_config
from metabric_survival.data import load_dataset


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def ds(cfg):
    return load_dataset(cfg)
