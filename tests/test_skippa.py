#!/usr/bin/env python

"""Tests for `skippa` package."""

import pytest

import pandas as pd

from skippa import Skippa, columns


def test_pipeline(test_data):
    """Test if the pipeline works, i.e. returns a dataframe"""
    X, _ = test_data
    pipe = (
        Skippa()
        .select(columns())
        .build()
    )
    res = pipe.fit_transform(X)
    assert isinstance(res, pd.DataFrame)
    assert X.shape == res.shape
