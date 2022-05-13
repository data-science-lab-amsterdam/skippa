"""
DataProfile is used for storing and retrieving metadata of data that is used in the pipeline.
Typically the DataProfile is created during fitting of a pipeline.
The profile is used by the Gradio app that can be created.
"""
from typing import Optional, Any, Dict, Generator

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_float_dtype

class DataProfile:

    MAX_NUM_DISTINCT_VALUES = 100000

    def __init__(self, df: pd.DataFrame, y: Optional[Any] = None) -> None:
        self.column_names = df.columns.tolist()
        self.dtypes = df.dtypes
        self.info = {}
        self.info_labels = {}
        self._profile_features(df)
        self._profile_labels(y)

    def _profile_features(self, df: pd.DataFrame) -> None:
        """Create a profile of the features"""
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
    
    def _profile_labels(self, y) -> None:
        """Create a profile of the labels (if present)"""
        if y is None:
            self.info_labels['type'] = None
        y = np.array(y)
        try:
            n_cols = y.shape[1]
        except IndexError:
            n_cols = 1
        
        if n_cols > 1:
            # assume multi-class classification
            self.info_labels['type'] = 'multi-class'
        else:
            if sorted(pd.Series(y).unique().astype('int')) == [0, 1]:
                self.info_labels['type'] = 'binary'
            else:
                self.info_labels['type'] = 'regression'

    def __iter__(self) -> Generator[Dict, None, None]:
        for column_name, info in self.info.items():
            info['name'] = column_name
            yield info
    
    def is_classification(self) -> bool:
        return self.info_labels['type'] in ['binary', 'multi-class']
    
    def is_regression(self) -> bool:
        return self.info_labels['type'] == 'regression'
