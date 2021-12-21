import pytest

from skippa import columns


def test_columns_unnamedarg(test_df):
    names = columns(['x'])(test_df)
    assert names == ['x']


def test_columns_columnselectorarg(test_df):
    selector = columns(['x'])
    names = columns(selector)(test_df)
    assert names == ['x']

def test_columns_include(test_df):
    selector = columns(include=['x', 'y'])
    names = selector(test_df)
    assert names == ['x', 'y']

def test_columns_exclude(test_df):
    selector = columns(exclude=['x', 'y'])
    names = selector(test_df)
    assert set(names) & set(['x', 'y']) == set()
