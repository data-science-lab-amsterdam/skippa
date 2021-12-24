"""
DataProfile is used for storing and retrieving metadata
"""
from typing import Optional, Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_float_dtype

class DataProfile:

    MAX_NUM_DISTINCT_VALUES = 100000

    def __init__(self, df: pd.DataFrame, y: Optional[Any] = None):
        self.column_names = df.columns.tolist()
        self.dtypes = df.dtypes
        self.info = {}

        for column_name, dtype in zip(self.column_names, self.dtypes):
            column_info = {
                'dtype': dtype
            }
            if is_numeric_dtype(dtype):
                column_info['is_numeric'] = True
                column_info['is_string'] = False
                column_info['min_value'] = df[column_name].min()
                column_info['max_value'] = df[column_name].max()
                column_info['median_value'] = df[column_name].median()
            elif is_string_dtype(dtype):
                column_info['is_numeric'] = False
                column_info['is_string'] = True
                if df[column_name].nunique() <= self.MAX_NUM_DISTINCT_VALUES:
                    column_info['values'] = df[column_name].unique().tolist()
                else:
                    column_info['values'] = []
                column_info['mode'] = df[column_name].value_counts().idxmax()
            else:
                raise ValueError(f'No profile for column {column_name}')

            self.info[column_name] = column_info

    def __iter__(self):
        for column_name, info in self.info.items():
            info['name'] = column_name
            yield info
