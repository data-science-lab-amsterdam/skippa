"""
Zie https://scikit-learn.org/stable/modules/compose.html

Stage(steps=[
    Step(
        name='impute',
        cols=Cols(by_dtype='numeric')),
        transformation=SimpleImputer(type='median')
    )
])


df = pd.read_csv(...)

df_transformed = (
    SklearnPreprocessor(df)
    .impute(Cols(dtype_include='numeric'),  strategy='median')
    .scale(Cols(dtype_include='numeric'), type='simple')
)


"""
from __future__ import annotations

from typing import Optional, Union, List, Callable
from dataclasses import dataclass, field
import re

from copy import deepcopy
import dill
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression

from preprocessing.transformers import (
    XSimpleImputer,
    XStandardScaler,
    XMinMaxScaler,
    XOneHotEncoder,
    xmake_column_transformer,
    ColumnSelector,
    columns
)


class PandasTransformer(BaseEstimator, TransformerMixin):
    """Transform numpy array to pandas dataframe."""

    def __init__(self, current_columns: List[str], transformed_column: str, callback: Callable, **kwargs) -> None:
        self.current_columns = current_columns
        self.transformed_column = transformed_column
        self.callback = callback
        #super().__init__(**kwargs)

    def fit(self, X, **kwargs):
        print(self.current_columns)
        n_columns = X.shape[1]
        n_new_columns = n_columns - len(self.current_columns) + 1
        self.new_columns = [
            c for c in self.current_columns if c != self.transformed_column
        ] + [
            f'{self.transformed_column}_{i}' for i in range(n_new_columns)
        ]
        print('fit', self.transformed_column, self.new_columns)
        return self

    def transform(self, X, y = None, **kwargs):
        print('transform', self.new_columns)
        self.callback(self.transformed_column, self.new_columns)
        return pd.DataFrame(data=X, columns=self.new_columns)


class SklearnPreprocessor:

    def __init__(self) -> None:
        self.pipeline_steps = []
        # self._df = df.copy().iloc[[0], :]
        # self._columns = df.columns.values
        self._step_idx: int = 0

    def build(self, **kwargs):
        return Pipeline(steps=self.pipeline_steps, **kwargs)

    @staticmethod
    def load_pipeline(path) -> Pipeline:
        with open(path, 'rb') as f:
            pipe = dill.loads(f.read())
            return pipe

    def _evaluate_columns(self, cols: ColumnSelector) -> List[str]:
        """Evaluate a column selection expression

        Args:
            cols (ColumnSelector): [description]

        Returns:
            List[str]: [description]
        """
        return cols(self._df)

    def _column_update(self, transformed_column: str, new_columns: List[str]) -> None:
        print('new columns: ', new_columns)
        added_columns = [c for c in new_columns if c not in self._columns]
        self._columns = new_columns
        new_df = self._df.drop(transformed_column, axis=1)
        new_df = new_df.assign(**{c: None for c in added_columns})
        self._df = new_df
        print(self._df)

    def _step(
        self,
        name: str,
        transformation: Transformation,
        cols: Optional[ColumnSelector] = None,
        transformed_cols: Optional[Callable] = None
    ) -> None:
        name = f'{name}_{self._step_idx}'
        if cols is not None:
            inner_name = f'{cols.name}_{self._step_idx}'
            transformer = xmake_column_transformer(
                (transformation, cols),
                remainder='passthrough'
            )
            self.pipeline_steps.append(
                (inner_name, transformer)
            )
            # for column_name in self._evaluate_columns(cols):
            #     inner_name = f'{cols.name}_{column_name}_{self._step_idx}'
            #     transformer = make_column_transformer(
            #         (transformation, [column_name]),
            #         remainder='passthrough'
            #     )
            #     self.pipeline_steps.append(
            #         (inner_name, transformer)
            #     )
        else:
            self.pipeline_steps.append((name, transformation))

        self._step_idx += 1

    def impute(self, cols: ColumnSelector, **kwargs) -> SklearnPreprocessor:
        #self._step(name='impute', cols=cols, transformation=XSimpleImputer(**kwargs))
        self.pipeline_steps.append(('impute', XSimpleImputer(cols=cols, **kwargs)))
        return self

    def scale(self, cols: ColumnSelector, type: str = 'standard', **kwargs) -> SklearnPreprocessor:
        if type == 'standard':
            transformation = XStandardScaler(cols=cols, **kwargs)
        elif type == 'minmax':
            transformation = XMinMaxScaler(cols=cols, **kwargs)
        else:
            raise ValueError(f'Invalid scaler type "{type}". Choose standard or minmax.')
        #self._step(name=f'scale_{type}', cols=cols, transformation=transformation, transformed_cols=lambda c: f'{c}_sc')
        self.pipeline_steps.append((f'scale_{type}', transformation))
        return self

    def onehot(self, cols: ColumnSelector, **kwargs) -> SklearnPreprocessor:
        if cols is None:
            cols = columns(dtype_include='category')

        kwargs['sparse'] = False
        self.pipeline_steps.append(('onehot', XOneHotEncoder(cols=cols, **kwargs)))
        return self

    def select(self, cols: Union[ColumnSelector, List[str]]) -> SklearnPreprocessor:
        # if not isinstance(cols, Cols):
        #     cols = Cols(include=cols)
        if isinstance(cols, ColumnSelector):
            cols = cols()
        transformation = ColumnTransformer(
            [("selector", "passthrough", cols)],
            remainder="drop"
        )
        self._step(name='select', transformation=transformation)

        return self

    def add(self, cols: Union[ColumnSelector, List[str]]) -> SklearnPreprocessor:
        self._step(name='add', transformation=(cols, None))
        return self

