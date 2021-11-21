import pytest

from skippa import columns


def test_columns_unnamedarg(test_data):
    names = columns(['x'])(test_data)
    assert names == ['x']


def test_columns_columnselectorarg(test_data):
    selector = columns(['x'])
    names = columns(selector)(test_data)
    assert names == ['x']

def test_columns_include(test_data):
    selector = columns(include=['x', 'y'])
    names = selector(test_data)
    assert names == ['x', 'y']

def test_columns_exclude(test_data):
    selector = columns(exclude=['x', 'y'])
    names = selector(test_data)
    assert set(names) & set(['x', 'y']) == set()
