#!/usr/bin/env python

"""Tests for `skippa` package."""

import pytest

import pandas as pd

from skippa import Skippa, columns


def test_pipeline(test_data):
    """Test if the pipeline works, i.e. returns a dataframe"""
    pipe = (
        Skippa()
        .select(columns())
        .build()
    )
    res = pipe.fit_transform(test_data)
    assert isinstance(res, pd.DataFrame)
    assert test_data.shape == res.shape
