from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple, Callable
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import make_column_selector, make_column_transformer


class ColumnSelector:
    """This is not a transformer, but a utility class for defining a column set."""

    def __init__(self, selector: Callable) -> None:
        self.selector = selector
        self.name = re.sub('[^a-zA-Z0-9_]', '', f'select_{selector}')

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """A ColumnsSelector can be called on a dataframe

        Args:
            df (pd.DataFrame): pandas df

        Returns:
            List[str]: A list of column names
        """
        return self.selector(df)

    def __add__(self, other: ColumnSelector) -> ColumnSelector:
        """Add two selectors.

        N.B. Adding means taking the intersection because we don't want duplicates.
        In order to preserve the order in existing selectors, the use of set is avoided.

        Args:
            other (ColumnSelector): Another column selector

        Returns:
            ColumnSelector: A new one with merged selector callables
        """
        assert isinstance(other, ColumnSelector), 'Argument should be of type ColumnSelector'
        def _union_preserving_order(list1, list2):
            intersection = set(list1) & set(list2)
            return list1 + [x for x in list2 if x not in intersection]

        return ColumnSelector(
            lambda df: _union_preserving_order(self.__call__(df), other(df))
        )

    def __sub__(self, other: ColumnSelector) -> ColumnSelector:
        """Not sure if this is ever practical, but if you make an __add__...

        Args:
            other (ColumnSelector): Another column selector

        Returns:
            ColumnSelector: [description]
        """
        assert isinstance(other, ColumnSelector), 'Argument should be of type ColumnSelector'
        return ColumnSelector(lambda df: [c for c in self.__call__(df) if c not in other(df)])

    def __str__(self) -> str:
        """Simple string representation

        Returns:
            str: This string is shown as a name in pipeline steps
        """
        return self.name


# New types for use in type annotation
Transformation = Union[BaseEstimator, TransformerMixin]
ColumnExpression = Union[ColumnSelector, List[str]]


def columns(
    *args,
    include: Optional[ColumnExpression] = None,
    exclude: Optional[ColumnExpression] = None,
    **kwargs
) -> ColumnSelector:
    """Helper function for creating a ColumnSelector

    Flexible arguments:
    - include or exclude lists: speak for themselves
    - dtype_include, dtype_exclude, pattern: dispatched to sklearn's make_column_selector
    - otherwise: a list to include, or an existing ColumnSelector

    Args:
        include (Optional[ColumnExpression], optional): [description]. Defaults to None.
        exclude (Optional[ColumnExpression], optional): [description]. Defaults to None.

    Returns:
        ColumnSelector: A callable that returns columns names, when called on a df
    """
    if len(args) == 1:
        include = args[0]

    if isinstance(include, ColumnSelector):
        return include

    if include is not None:
        #selector = (lambda df: [c for c in df.columns if c in include])
        #selector = (lambda df: [c for c in include if c in df.columns])
        selector = lambda _: list(include)
    elif exclude is not None:
        selector = (lambda df: [c for c in df.columns if c not in exclude])
    else:
        selector = make_column_selector(**kwargs)
    return ColumnSelector(selector)


class XMixin:
    """Utility class providing additional methods for custom Skippa transformers."""
    
    def _set_columns(self, cols: ColumnSelector) -> None:
        self.cols = cols

    def _evaluate_columns(self, X):
        self._column_names = self.cols(X)
        return self._column_names

    def _get_result(self, X, res) -> pd.DataFrame:
        column_names = self._evaluate_columns(X)
        X.loc[:, column_names] = res
        return X

    def _set_names(self, X):
        self.names = X.columns

    def _get_names(self):
        return self.names


class XColumnTransformer(ColumnTransformer, XMixin):
    """Custom ColumnTransformer. Probably not needed anymore."""

    def fit(self, X, y=None, **kwargs):
        self._set_names(X)
        super().fit(X, **kwargs)
        return self

    def transform(self, X, y=None):
        res = super().transform(X=X)
        return pd.DataFrame(data=res, columns=self._get_names())

    def fit_transform(self, X, y=None):
        self._set_names(X)
        res = super().fit_transform(X=X, y=None)
        return pd.DataFrame(data=res, columns=self._get_names())


def xmake_column_transformer(
    *transformers,
    remainder="drop",
    **kwargs
):
    """Custom wrapper around sklearn's make_column_transformer"""
    transformers, columns = zip(*transformers)
    names = [str(t) for t in transformers]

    transformer_list = list(zip(names, transformers, columns))
    return XColumnTransformer(
        transformer_list,
        remainder=remainder,
        **kwargs
    )


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


class XSimpleImputer(SimpleImputer, XMixin):
    """Wrapper round sklearn's SimpleImputer"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        res = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = res
        return df


class XStandardScaler(StandardScaler, XMixin):
    """Wrapper round sklearn's StandardScaler"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        res = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = res
        return df


class XMinMaxScaler(MinMaxScaler, XMixin):
    """Wrapper round sklearn's MinMaxScaler"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        res = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = res
        return df


class XOneHotEncoder(OneHotEncoder, XMixin):
    """Wrapper round sklearn's OneHotEncoder"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        data_new = super().transform(X[column_names], **kwargs)
        df_new = X.drop(column_names, axis=1)
        new_column_names = self.get_feature_names_out()
        df_new.loc[:, new_column_names] = data_new
        return df_new


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
