import pytest
import pandas as pd


collect_ignore = ['setup.py']

@pytest.fixture
def test_data():
    return pd.DataFrame({
        'q': [2, 3, 4],
        'x': ['a', 'b', 'c'],
        'y': [1, 16, 1000],
        'z': [0.4, None, 8.7]
    })

