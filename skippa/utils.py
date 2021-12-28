from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def get_dummy_data(
    nrows: int = 100,
    nfloat: int = 4,
    nint: int = 2,
    nchar: int = 3,
    ndate: int = 1,
    missing: bool = True,
    binary_y: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create dummy data.

    Args:
        nrows (int, optional): Number of total rows. Defaults to 100.
        nfloat (int, optional): Number of float columns. Defaults to 4.
        nint (int, optional): Number of integer columns. Defaults to 2.
        nchar (int, optional): Number of character/categorical columns. Defaults to 3.
        ndate (int, optional): Number of date columns. Defaults to 1.
        binary_y (bool, optional): If True, returns 0's & 1's for y, otherwise float values between 0 & 100

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A pandas DataFrame for features and a numpy array for labels
    """
    n_total_columns = nfloat + nint + nchar + ndate
    columns = [chr(i) for i in range(97, 97 + n_total_columns)]
    data = pd.DataFrame(columns=columns)
    idx = 0

    # float columns
    idx_start_float = idx
    if nfloat > 0:
        data[columns[idx:idx+nfloat]] = np.random.random((nrows, nfloat)) * np.arange(1, nfloat+1)
        idx += nfloat

    # int columns
    idx_start_int = idx
    if nint > 0:
        data[columns[idx:idx+nint]] = np.random.randint(10, size=(nrows, nint))
        idx += nint

    # char columns
    idx_start_char = idx
    start = 97
    n_distinct_values = 4
    for i in range(nchar):
        values = [chr(i) for i in np.random.randint(n_distinct_values, size=nrows) + start]
        data.iloc[:, idx] = pd.Series(values)
        start += n_distinct_values
        idx += 1
    
    # date columns
    idx_start_date = idx
    for i in range(ndate):
        dates = [datetime.now() + timedelta(days=-i) for i in range(nrows)]
        data.iloc[:, idx] = pd.Series([f'{d:%Y-%m-%d}' for d in dates])
        idx += 1

    # set missing values
    if missing:
        if nfloat > 0:
            data.iloc[0, idx_start_float] = np.nan
        if nint > 0:
            data.iloc[1, idx_start_int] = np.nan
        if nchar > 0:
            data.iloc[2, idx_start_char] = None

    if binary_y:
        y = (np.random.random((nrows,)) >= 0.75) * 1.
    else:
        y = np.random.random((nrows,)) * 100.
    return data, y
