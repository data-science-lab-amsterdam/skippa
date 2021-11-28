"""
This implements transformers based on existing sklearn transformers
"""
from typing import Optional, Union, List, Dict, Tuple, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from skippa.transformers import ColumnSelector, XMixin


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
) -> XColumnTransformer:
    """Custom wrapper around sklearn's make_column_transformer"""
    transformers, columns = zip(*transformers)
    names = [str(t) for t in transformers]

    transformer_list = list(zip(names, transformers, columns))
    return XColumnTransformer(
        transformer_list,
        remainder=remainder,
        **kwargs
    )


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
