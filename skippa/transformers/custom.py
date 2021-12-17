"""
This defines custom transformers implementing anything other than 
existing skleafrn treansformers.
"""
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skippa.transformers import ColumnSelector, SkippaMixin


class SkippaApplier(BaseEstimator, TransformerMixin, SkippaMixin):
    """Transformer for applying arbitrary function (wraps around pandas apply)"""

    def __init__(self, cols: ColumnSelector, *args, **kwargs):
        """Initialise with columns specifier and apply parameters

        Args:
            cols (ColumnSelector): columns specifier
            *args, **kwargs: any arguments accepted by pandas.DataFrame.apply()
        """
        self.cols = cols
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y=None, **fit_params):
        """Nothing to do here"""
        return self

    def transform(self, X, y=None, **transform_params):
        """Use pandas.DataFrame.apply method"""
        column_names = self._evaluate_columns(X)
        data_new = X[column_names].apply(*self.args, **self.kwargs)
        if not isinstance(data_new, pd.DataFrame):
            raise TypeError('Applied function should return a pandas dataframe!')
        df_new = X.drop(column_names, axis=1)
        df_new.loc[:, column_names] = data_new
        return df_new


class SkippaCaster(BaseEstimator, TransformerMixin, SkippaMixin):
    """Transformer for casting columns to another data type"""

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


class SkippaRenamer(BaseEstimator, TransformerMixin):
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
        evaluate the column names and apply the renaming function
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


class SkippaSelector(BaseEstimator, TransformerMixin, SkippaMixin):
    """Transformer for selecting a subset of columns in a df."""

    def __init__(self, cols: ColumnSelector) -> None:
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        df = X.copy()
        return df[column_names]


class SkippaAssigner(BaseEstimator, TransformerMixin, SkippaMixin):
    """Transformer for selecting a subset of columns in a df."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        df = X.assign(**self.kwargs)
        return df


class SkippaConcat(BaseEstimator, SkippaMixin):
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


class SkippaDateFormatter(BaseEstimator, TransformerMixin, SkippaMixin):
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


class SkippaDateEncoder(BaseEstimator, TransformerMixin, SkippaMixin):
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


class SkippaOutlierRemover(BaseEstimator, TransformerMixin, SkippaMixin):
    """Detect and remove outliers, based on simple IQR"""

    def __init__(self, cols: ColumnSelector, factor: float = 1.5):
        self.cols = cols
        self.factor = factor
        self.statistics = {}
        
    @staticmethod
    def _get_iqr_range(x: pd.Series, factor: float = 1.5) -> Tuple[float, float]:
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        return lower_bound, upper_bound

    @staticmethod
    def _limit(x: pd.Series, bounds: Tuple[float, float]) -> pd.Series:
        x_ = x.copy()
        lower, upper = bounds
        x_[(x_ < lower) | (x_ > upper)] = np.nan
        return x_

    def fit(self, X, y=None):
        for column_name in self._evaluate_columns(X):
            x = X[column_name]
            lower, upper = self._get_iqr_range(x, factor=self.factor)
            self.statistics[column_name] = (lower, upper)
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        for column_name in self._evaluate_columns(X):
            df[column_name] = df[column_name].apply(lambda x: self._limit(x, bounds=self.statistics[column_name]))
        return df

