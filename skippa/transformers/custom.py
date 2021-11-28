"""
This defines custom transformers implementing anything other than 
existing skleafrn treansformers.
"""
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skippa.transformers import ColumnSelector, XMixin


class XCaster(BaseEstimator, TransformerMixin, XMixin):
    """Transformer for renaming columns"""

    def __init__(self, cols: ColumnSelector, dtype: Any) -> None:
        """There are 2 ways to define a mapping for renaming

        - a dict of old: new mappings
        - a column selector and a renaming fuction

        Args:
            dtype (Any): Either a single dtype, of a dict mapping column to dtype
        """
        self.cols = cols
        self.dtype = dtype

    def fit(self, X, y=None, **kwargs):
        """Nothing to do here."""
        return self

    def transform(self, X, y=None, **kwargs):
        """Apply the actual casting using pandas.astype"""
        column_names = self._evaluate_columns(X)
        df = X.copy()
        df[column_names] = df[column_names].astype(self.dtype)
        return df


class XRenamer(BaseEstimator, TransformerMixin):
    """Transformer for renaming columns"""

    def __init__(self, mapping: Union[Dict, Tuple[ColumnSelector, Callable]]) -> None:
        """There are 2 ways to define a mapping for renaming

        - a dict of old: new mappings
        - a column selector and a renaming fuction

        Args:
            mapping (Union[Dict, Tuple[ColumnSelector, Callable]]): [description]
        """
        self.mapping = mapping

    def fit(self, X, y=None, **kwargs):
        """Look at the df to determine the mapping.
        
        In case of a columnselector + function: 
        evaluate the column names and aplpy the renaming function
        """
        if isinstance(self.mapping, tuple):
            column_selector, renamer = self.mapping
            column_names = column_selector(X)
            self.mapping_dict = {c: renamer(c) for c in column_names}
        else:
            self.mapping_dict = self.mapping
        return self

    def transform(self, X, y=None, **kwargs):
        """Apply the actual renaming using pandas.rename"""
        df_renamed = X.rename(self.mapping_dict, axis=1)
        return df_renamed


class XSelector(BaseEstimator, TransformerMixin, XMixin):
    """Transformer for selecting a subset of columns in a df."""

    def __init__(self, cols: ColumnSelector) -> None:
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        df = X.copy()
        return df[column_names]


class XAssigner(BaseEstimator, TransformerMixin, XMixin):
    """Transformer for selecting a subset of columns in a df."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        df = X.assign(**self.kwargs)
        return df




class XConcat(BaseEstimator, XMixin):
    """Concatenate two pipelines."""

    def __init__(self, left, right) -> None:
        self.name1, self.pipe1 = left
        self.name2, self.pipe2 = right

    def fit(self, X, y=None, **kwargs):
        self.pipe1.fit(X=X, y=y, **kwargs)
        self.pipe2.fit(X=X, y=y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        df1 = self.pipe1.transform(X, **kwargs)
        df2 = self.pipe2.transform(X, **kwargs)
        return pd.concat([df1, df2], axis=1)


class XDateFormatter(BaseEstimator, TransformerMixin, XMixin):
    """Data strings into pandas datetime"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        """Nothing to do here"""
        return self

    def transform(self, X, y=None, **kwargs):
        """Apply the transformation"""
        column_names = self._evaluate_columns(X)
        df = X.copy()
        df[column_names] = df[column_names].apply(pd.to_datetime)
        return df


class XDateEncoder(BaseEstimator, TransformerMixin, XMixin):
    """Derive date features using pandas datatime's .dt property."""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        parts = {'year': True, 'month': True, 'day': True}
        parts.update(kwargs)
        self._parts = parts
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        df = X.copy()
        for c in column_names:
            dt = X[c].apply(pd.to_datetime).dt
            date_part_values = [getattr(dt, part) for part in self._parts]
            res = pd.concat(date_part_values, axis=1)
            res.columns = [f'{c}_{part}' for part in self._parts]
            df = df.drop([c], axis=1)
            df = pd.concat([df, res], axis=1)
        return df


class OutlierRemover(BaseEstimator,TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        
    def _outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self._outlier_detector)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        return X
