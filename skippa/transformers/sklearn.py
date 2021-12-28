"""
This implements transformers based on existing sklearn transformers
"""
from typing import Optional, Union, List, Dict, Tuple, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    FunctionTransformer
)
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from skippa.transformers import ColumnSelector, SkippaMixin


class SkippaColumnTransformer(ColumnTransformer, SkippaMixin):
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


def make_skippa_column_transformer(
    *transformers,
    remainder="drop",
    **kwargs
) -> SkippaColumnTransformer:
    """Custom wrapper around sklearn's make_column_transformer"""
    transformers, columns = zip(*transformers)
    names = [str(t) for t in transformers]

    transformer_list = list(zip(names, transformers, columns))
    return SkippaColumnTransformer(
        transformer_list,
        remainder=remainder,
        **kwargs
    )


class SkippaSimpleImputer(SkippaMixin, SimpleImputer):
    """Wrapper round sklearn's SimpleImputer"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        if kwargs.get('strategy', 'mean') == 'most_frequent':
            self._dtype_required = 'string'
            kwargs['missing_values'] = None
        else:
            self._dtype_required = 'numeric'
            kwargs['missing_values'] = np.nan

        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes=self._dtype_required)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes=self._dtype_required)
        res = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = res
        return df


class SkippaStandardScaler(SkippaMixin, StandardScaler):
    """Wrapper round sklearn's StandardScaler"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes='numeric')
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes='numeric')
        res = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = res
        return df


class SkippaMinMaxScaler(SkippaMixin, MinMaxScaler):
    """Wrapper round sklearn's MinMaxScaler"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes='numeric')
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes='numeric')
        res = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = res
        return df


class SkippaOneHotEncoder(SkippaMixin, OneHotEncoder):
    """Wrapper round sklearn's OneHotEncoder"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        kwargs['sparse'] = False  # never output a sparse matrix
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
        assert len(new_column_names) == data_new.shape[1], "Nr. of expected vs. actual columns doesn't match"
        df_new.loc[:, new_column_names] = data_new
        return df_new


class SkippaOrdinalEncoder(SkippaMixin, OrdinalEncoder):
    """Wrapper round sklearn's OrdinalEncoder"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        encoded = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = encoded
        return df


class SkippaLabelEncoder(SkippaMixin, LabelEncoder):
    """Wrapper round sklearn's LabelEncoder"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X)
        encoded = super().transform(X[column_names], **kwargs)
        df = X.copy()
        df.loc[:, column_names] = encoded
        return df


class SkippaPCA(SkippaMixin, PCA):
    """Wrapper round sklearn's PCA"""

    def __init__(self, cols: ColumnSelector, **kwargs) -> None:
        self._set_columns(cols)
        super().__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes='number')
        print(column_names)
        super().fit(X[column_names], **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        column_names = self._evaluate_columns(X, check_dtypes='number')
        data_new = super().transform(X[column_names], **kwargs)
        df_new = X.drop(column_names, axis=1)
        new_column_names = [f'c{i}' for i in range(self.n_components_)]
        assert len(new_column_names) == data_new.shape[1], "Nr. of expected vs. actual columns doesn't match"
        df_new.loc[:, new_column_names] = data_new
        return df_new
    
    def fit_transform(self, X, y=None, **kwargs):
        """The PCA parent class has a custom .fit_transform method for some reason."""
        column_names = self._evaluate_columns(X, check_dtypes='number')
        pca_result = super().fit_transform(X[column_names], y, **kwargs)
        df_new = X.drop(column_names, axis=1)
        new_column_names = [f'c{i}' for i in range(self.n_components_)]
        assert len(new_column_names) == pca_result.shape[1], "Nr. of expected vs. actual columns doesn't match"
        df_new.loc[:, new_column_names] = pca_result
        return df_new
