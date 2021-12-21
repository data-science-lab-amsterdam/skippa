import pytest
import numpy as np
import pandas as pd

from skippa.utils import get_dummy_data


collect_ignore = ['setup.py']


@pytest.fixture
def test_data():
    return get_dummy_data(nrows=10)


@pytest.fixture
def test_df(test_data):
    X, _ = test_data
    return X


@pytest.fixture
def test_data_large():
    return get_dummy_data(nrows=200)
