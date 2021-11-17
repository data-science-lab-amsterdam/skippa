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


def enrich(transformer: Transformation) -> Transformation:
    transformer = deepcopy(transformer)
    def enriched_fit(X, y=None, **kwargs):
        print('fit!')
        return transformer.fit(X=X, y=y, **kwargs)
    #transformer.fit = enriched_fit
    setattr(transformer, 'fit', enriched_fit)
    return transformer

# est = enrich(StandardScaler)()
# est.fit_transform(df[['y', 'z']])

est = StandardScaler()
est.fit_transform(df[['y']])


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
        assert isinstance(other, ColumnSelector), 'Argument should be of type ColumnSelector'
        return ColumnSelector(lambda df: list(set(self.__call__(df) + other(df))))

    def __sub__(self, other):
        assert isinstance(other, ColumnSelector), 'Argument should be of type ColumnSelector'
        return ColumnSelector(lambda df: [c for c in self.__call__(df) if c not in other(df)])

    def __str__(self):
        return self.name


class BackToPandas(BaseEstimator, TransformerMixin):

    def __init__(self, columns, **kwargs) -> None:
        self.columns = columns
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(data=X, columns=self.columns)


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

    def __init__(self, df: pd.DataFrame) -> None:
        self.pipeline_steps = []
        self._df = df.copy().iloc[[0], :]
        self._columns = df.columns.values
        self._step_idx: int = 0

    def build(self, **kwargs):
        return Pipeline(steps=self.pipeline_steps, **kwargs)
        # return DataFrameMapper(
        #     [transformation for (name, transformation) in self.pipeline_steps],
        #     df_out=False,
        #     default=None
        # )

    @staticmethod
    def _pandarizer(current_columns: List[str], transformed_column: str, callback: Callable) -> FunctionTransformer:
        n_current_columns = len(current_columns)

        # def _np_to_pd(data, y=None, **kwargs):
        #     n_columns = data.shape[1]
        #     n_new_columns = n_columns - n_current_columns
        #     new_columns = [
        #         c for c in current_columns if c != transformed_column
        #     ] + [
        #         f'{transformed_column}_{i}' for i in range(n_new_columns)
        #     ]
        #     callback(new_columns)
        #     return pd.DataFrame(data=data, columns=new_columns)

        # return FunctionTransformer(_np_to_pd)
        return PandasTransformer(current_columns, transformed_column, callback)
        
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

    @staticmethod
    def _creates_multiple_columns(transformation: Transformation) -> bool:
        transformers_creating_multiple_columns = [OneHotEncoder, DateEncoder]
        return any([
            isinstance(transformation, cls)
            for cls in transformers_creating_multiple_columns
        ])

    def _step(
        self,
        name: str,
        transformation: Transformation,
        cols: Optional[ColumnSelector] = None,
        transformed_cols: Optional[Callable] = None
    ) -> None:
        name = f'{name}_{self._step_idx}'
        if cols is not None:
            for column_name in self._evaluate_columns(cols):
                inner_name = f'{cols.name}_{column_name}_{self._step_idx}'
                transformer = make_column_transformer(
                    (transformation, [column_name]),
                    remainder='passthrough'
                )
                self.pipeline_steps.append(
                    (inner_name, transformer)
                )
                pandarizer = PandasTransformer(
                    current_columns=self._columns,
                    transformed_column=column_name,
                    callback=self._column_update
                )
                self.pipeline_steps.append(
                    (f'{inner_name}_pd', pandarizer)
                )
            
        else:
            self.pipeline_steps.append((name, transformation))

        self._step_idx += 1

    def impute(self, cols: ColumnSelector, **kwargs) -> SklearnPreprocessor:
        self._step(name='impute', cols=cols, transformation=SimpleImputer(**kwargs))
        return self

    def scale(self, cols: ColumnSelector, type: str, **kwargs) -> SklearnPreprocessor:
        if type == 'standard':
            transformation = StandardScaler(**kwargs)
        elif type == 'minmax':
            transformation = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f'Invalid scaler type "{type}". Choose standard or minmax.')
        self._step(name=f'scale_{type}', cols=cols, transformation=transformation, transformed_cols=lambda c: f'{c}_sc')
        return self

    def onehot(self, cols: ColumnSelector, **kwargs) -> SklearnPreprocessor:
        if cols is None:
            cols = columns(dtype_include='category')

        kwargs['sparse'] = False
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
    SklearnPreprocessor(df)
        .impute(columns(dtype_include='number'), strategy='median')
        #.scale(columns(dtype_include='number'), type='standard')
        #.onehot(columns(['x']))
        #.select([0, 1, 2])
        #.select(['x', 'z'])
        .build(verbose=True)
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
        (['x'], OneHotEncoder())
    ],
    input_df=True,
    df_out=True,
    default=None
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
