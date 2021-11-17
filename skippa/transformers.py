from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
import re
from copy import deepcopy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import make_column_selector, make_column_transformer


class ColumnSelector:

    def __init__(self, selector: Callable) -> None:
        self.selector = selector
        self.name = re.sub('[^a-zA-Z0-9_]', '', f'select_{selector}')

    def __call__(self, df) -> List:
        return self.selector(df)

    def __add__(self, other):
        assert isinstance(other, ColumnSelector), 'Argument should be of type ColumnSelector'
        return ColumnSelector(lambda df: list(set(self.__call__(df) + other(df))))

    def __sub__(self, other):
        assert isinstance(other, ColumnSelector), 'Argument should be of type ColumnSelector'
        return ColumnSelector(lambda df: [c for c in self.__call__(df) if c not in other(df)])

    def __str__(self):
        return self.name


Transformation = Union[BaseEstimator, TransformerMixin]
ColumnExpression = Union[ColumnSelector, List[str]]


def columns(
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **kwargs
) -> ColumnSelector:
    if isinstance(include, ColumnSelector):
        return include

    if include is not None:
        selector = (lambda df: [c for c in df.columns if c in include])
    elif exclude is not None:
        selector = (lambda df: [c for c in df.columns if c not in exclude])
    else:
        selector = make_column_selector(**kwargs)
    return ColumnSelector(selector)





class XMixin:
    
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

    def fit(self, X, y=None, **kwargs):
        self._set_names(X)
        super().fit(X, **kwargs)
        return self

    def transform(self, X):
        res = super().transform(X=X,)
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
    transformers, columns = zip(*transformers)
    names = [str(t) for t in transformers]

    transformer_list = list(zip(names, transformers, columns))
    return XColumnTransformer(
        transformer_list,
        remainder=remainder,
        **kwargs
    )


class XRenamer(BaseEstimator, TransformerMixin):

    def __init__(self, mapping: Union[Dict, Tuple[ColumnSelector, Callable]]) -> None:
        self.mapping = mapping

    def fit(self, X, **kwargs):
        if isinstance(self.mapping, tuple):
            column_selector, renamer = self.mapping
            column_names = column_selector(X)
            self.mapping_dict = {c: renamer(c) for c in column_names}
        else:
            self.mapping_dict = self.mapping
        return self

    def transform(self, X, **kwargs):
        df_renamed = X.rename(self.mapping_dict, axis=1)
        return df_renamed

class XSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cols: ColumnSelector) -> None:
        self.cols = cols

    def fit(self, X, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X[self.cols(X)]


class XSimpleImputer(SimpleImputer, XMixin):

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        res = super().transform(X[column_names], **kwargs)
        X = X.copy()
        X.loc[:, column_names] = res
        return X


class XStandardScaler(StandardScaler, XMixin):

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        res = super().transform(X[column_names], **kwargs)
        X = X.copy()
        X.loc[:, column_names] = res
        return X


class XMinMaxScaler(MinMaxScaler, XMixin):

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        res = super().transform(X[column_names], **kwargs)
        X = X.copy()
        X.loc[:, column_names] = res
        return X


class XOneHotEncoder(OneHotEncoder, XMixin):

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, **kwargs):
        column_names = self._evaluate_columns(X)
        data_new = super().transform(X[column_names], **kwargs)
        df_new = X.drop(column_names, axis=1)
        new_column_names = self.get_feature_names_out()
        df_new.loc[:, new_column_names] = data_new
        return df_new


class DateFormatter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate


class DateEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = X.apply(pd.to_datetime).dt
        return pd.concat([dt.year, dt.month, dt.day], axis=1)
