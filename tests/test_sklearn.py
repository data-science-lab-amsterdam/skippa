import pytest
import numpy as np
import pandas as pd

from skippa import columns
from skippa.transformers.sklearn import(
    SkippaSimpleImputer,
    SkippaStandardScaler,
    SkippaMinMaxScaler,
    SkippaOneHotEncoder,
    SkippaLabelEncoder,
    SkippaOrdinalEncoder,
    SkippaPCA
)
from skippa.utils import get_dummy_data


def test_simpleimputer_float(test_data):
    X, _ = test_data
    col_spec = columns(dtype_include='float')
    si = SkippaSimpleImputer(cols=col_spec, strategy='median')
    res = si.fit_transform(X)
    assert isinstance(res, pd.DataFrame)
    subset = res[col_spec(res)]
    assert subset.isna().sum().sum() == 0


def test_simpleimputer_int(test_data):
    X, _ = test_data
    col_spec = columns(dtype_include='int')
    si = SkippaSimpleImputer(cols=col_spec, strategy='median')
    res = si.fit_transform(X)
    assert isinstance(res, pd.DataFrame)
    subset = res[col_spec(res)]
    assert subset.isna().sum().sum() == 0


def test_simpleimputer_char(test_data):
    X, _ = test_data
    col_spec = columns(dtype_include='object')
    si = SkippaSimpleImputer(cols=col_spec, strategy='most_frequent')
    res = si.fit_transform(X)
    assert isinstance(res, pd.DataFrame)
    subset = res[col_spec(X)]
    assert subset.isna().sum().sum() == 0


def test_standardscaler():
    X, _ = get_dummy_data(nchar=0, ndate=0, nrows=10)
    ss = SkippaStandardScaler(cols=columns())
    res = ss.fit_transform(X)
    threshold = 0.01
    assert (np.abs(0 - res.mean()) < threshold).all()


def test_minmaxscaler():
    X, _ = get_dummy_data(nchar=0, ndate=0, nrows=10)
    mms = SkippaMinMaxScaler(cols=columns())
    res = mms.fit_transform(X)
    threshold = 0.01
    assert (np.abs(res.min() - 0.) < threshold).all()
    assert (np.abs(res.max() - 1.) < threshold).all()


def test_onehotencoder():
    X, _ = get_dummy_data(nrows=10, nfloat=0, nint=0, nchar=1, ndate=0)
    ohe = SkippaOneHotEncoder(cols=columns())
    res = ohe.fit_transform(X)
    n_distinct_values = X.iloc[:, 0].nunique(dropna=False)
    assert res.shape[1] == n_distinct_values


def test_pca():
    n_components = 3
    X, _ = get_dummy_data(nrows=100, nfloat=10, nint=0, nchar=1, ndate=0, missing=False)
    pca = SkippaPCA(cols=columns(dtype_include='float'), n_components=n_components)
    res = pca.fit_transform(X)
    assert pca.n_components_ == n_components
    assert res.shape[1] == n_components + 1
    expected_columns = [f'c{i}' for i in range(n_components)]
    assert all([c in res.columns for c in expected_columns])

