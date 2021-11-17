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

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper, gen_features

Transformation = Union[BaseEstimator, None]


def columns(
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **kwargs
) -> ColumnSelector:
    if include is not None:
        selector = (lambda df: [c for c in df.columns if c in include])
    elif exclude is not None:
        selector = (lambda df: [c for c in df.columns if c not in exclude])
    else:
        selector = make_column_selector(**kwargs)
    return ColumnSelector(selector)


class ColumnSelector:

    def __init__(self, selector: Callable) -> None:
        self.selector = selector
        self.name = re.sub('[^a-zA-Z0-9_]', '', f'select_{selector}')

    def __call__(self, df) -> List:
        return self.selector(df)

    def __add__(self, other):
        assert isinstance(other, ColumnSelector), 'Argument should be of type Cols'
        return ColumnSelector(lambda df: self.__call__(df) + other(df))

    def __sub__(self, other):
        assert isinstance(other, ColumnSelector), 'Argument should be of type Cols'
        return ColumnSelector(lambda df: [c for c in self.__call__(df) if c not in other(df)])

    def __str__(self):
        return self.name

# class PandasPipeline(Pipeline):

#     def fit(self, X, y=None, **kwargs):
#         assert isinstance(X, pd.DataFrame), f'Expected pandas dataframe, instead got {type(X)}'
#         self.columns = X.columns
#         super().fit(X=X, y=y, **kwargs)

#     def transform(self, X, y=None, **kwargs):
#         res = 
#         super().fit(X=X, y=y, **kwargs)


# class Cols:

#     def __init__(
#         self,
#         include: Optional[List[str]] = None,
#         exclude: Optional[List[str]] = None,
#         **kwargs
#     ) -> None:
#         self.include = include
#         self.exclude = exclude
#         self.kwargs = kwargs
#         self._columns = []
#         self.description = re.sub('[^a-zA-Z0-9_]', '', f'select_{kwargs}')

#     def __call__(self) -> Callable:
#         if self.include is not None:
#             return (lambda df: [c for c in df.columns if c in self.include])
#         elif self.exclude is not None:
#             return (lambda df: [c for c in df.columns if c not in self.exclude])
#         else:
#             return make_column_selector(**self.kwargs)

#     def __add__(self, other):
#         assert isinstance(other, Cols), 'Argument should be of type Cols'
#         #self.columns = self.columns + other()
#         return lambda: self.__call__() + other()

#     def __sub__(self, other):
#         assert isinstance(other, Cols), 'Argument should be of type Cols'
#         self.columns = [c for c in self.columns if c not in other()]


def pandarizer(selector: ColumnSelector) -> Transformer:
    return FunctionTransformer(lambda x: pd.DataFrame(x, columns = selector()))


class BackToPandas(BaseEstimator, TransformerMixin):

    def __init__(self, columns, **kwargs) -> None:
        self.columns = columns
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(data=X, columns=self.columns)


class SklearnPreprocessor:

    def __init__(self) -> None:
        self.pipeline_steps = []
        self._step_idx: int = 0

    def build(self, **kwargs):
        #return Pipeline(steps=self.pipeline_steps, **kwargs)
        return DataFrameMapper(
            [transformation for (name, transformation) in self.pipeline_steps],
            df_out=True,
            default=False
        )

    def _step(self, name: str, transformation: Transformation, cols: Optional[ColumnSelector] = None) -> None:
        name = f'{name}_{self._step_idx}'
        if cols is not None:
            #remainder_cols = columns() - cols
            inner_name = f'{cols.name}_{self._step_idx}'
            transformer = make_column_transformer(
                (transformation, cols),
                remainder='passthrough'
            )
            # self.pipeline_steps.append(
            #     (inner_name, transformer)
            # )
            self.pipeline_steps.append(
                (inner_name, (cols, transformation))
            )


            # transformer = make_column_transformer(
            #     (transformation, cols),
            #     remainder='drop'
            # )
            # remainder_cols = columns() - cols
            # remainder_transformer = make_column_transformer(
            #     ('passthrough', remainder_cols)
            # )
            # self.pipeline_steps.append(
            #     (inner_name, FeatureUnion([
            #         (f'{inner_name}_', transformer),
            #         (f'{inner_name}_remainder', remainder_transformer),
            #     ]))
            # )
            # self.pipeline_steps.append(
            #     (f'column_names_{self._step_idx}', pandarizer(cols() + remainder_cols()))
            # )
            
        else:
            self.pipeline_steps.append((name, transformation))

        #self.pipeline_steps.append(f'column_names_{self._step_idx}', pandarizer())

        self._step_idx += 1

    def impute(self, cols: ColumnSelector, **kwargs) -> SklearnPreprocessor:
        self._step(name='impute', cols=cols, transformation=SimpleImputer(**kwargs))
        return self

    def scale(self, cols: ColumnSelector, type: str, **kwargs) -> SklearnPreprocessor:
        if type == 'standard':
            t = StandardScaler(**kwargs)
        elif type == 'minmax':
            t = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f'Invalid scaler type "{type}". Choose standard or minmax.')
        self._step(name=f'scale_{type}', cols=cols, transformation=t)
        return self

    def onehot(self, cols: ColumnSelector, **kwargs) -> SklearnPreprocessor:
        if cols is None:
            cols = columns(dtype_include='category')
        
        self._step(
            name='onehot',
            transformation=OneHotEncoder(**kwargs),
            cols=cols
        )

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


df = pd.DataFrame({
    'x': ['a', 'b', 'c'],
    'y': [1, 16, 1000],
    'z': [0.4, None, 8.7]
})
pipe = (
    SklearnPreprocessor()
        .impute(columns(dtype_include='number'), strategy='median')
        #.scale(columns(dtype_include='number'), type='standard')
        #.onehot(columns)
        #.select([0, 1, 2])
        #.select(['x', 'z'])
        .build()
)

pipe.fit_transform(df)



pipe = (
    SklearnPreprocessor()
        .add('x')
        .impute(columns(['y', 'z']), strategy='median')
        #.impute(columns('z'), strategy='median')
        #.scale(columns(dtype_include='number'), type='standard')
        #.select([0, 1, 2])
        #.select(['x', 'z'])
        
        .build()
)

pipe = DataFrameMapper(
    features=[
        (['z'], SimpleImputer()),
        (['y'], StandardScaler()),
        (['x'], OneHotEncoder(), {'alias': 'q'}),
    ],
    input_df=True,
    df_out=True,
    default=False
)
pipe.fit_transform(df)



tmp = SimpleImputer(strategy='most_frequent')
tmp.fit_transform(df[['y', 'z']])
tmp.get_params()
SimpleImputer(strategy='most_frequent').fit_transform(df)

ohe = OneHotEncoder()
ohe.fit_transform(df[['x']])

ohe.get_feature_names()
# @dataclass
# class Step:

#     name: str
#     columns: Cols
#     transformation: Transformation





# Fetch Titanic dataset
titanic = fetch_openml('titanic', version=1, as_frame=True)

X = titanic.frame.drop('survived', axis=1)
y = titanic.frame['survived']

X.dtypes

# Scale numeric values
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# One-hot encode categorical values
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, make_column_selector(dtype_include='float64')),
        ('cat', cat_transformer, make_column_selector(dtype_include='category'))
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #('classifier', LogisticRegression())
])


clf.fit_transform(X)
clf.transform(X, y)
