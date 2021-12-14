"""
Defining a Skippa pipeline

>>> import pandas as pd
>>> from skippa import Skippa, columns
>>> from sklearn.linear_model import LogisticRegression


>>> X = pd.DataFrame({
>>>     'q': [2, 3, 4],
>>>     'x': ['a', 'b', 'c'],
>>>     'y': [1, 16, 1000],
>>>     'z': [0.4, None, 8.7]
>>> })
>>> y = np.array([0, 0, 1])

>>> pipe = (
>>>     Skippa()
>>>         .impute(columns(dtype_include='number'), strategy='median')
>>>         .scale(columns(dtype_include='number'), type='standard')
>>>         .onehot(columns(['x']))
>>>         .select(columns(['y', 'z']) + columns(pattern='x_*'))
>>>         .model(LogisticRegression())
>>> )

>>> pipe.fit(X=X, y=y)
>>> predictions = pipe.predict_proba(X)

"""
from __future__ import annotations

from typing import Any, Optional, Union, List, Callable, Type
from pathlib import Path

import dill
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.base import RegressorMixin, ClassifierMixin, ClusterMixin

from skippa.transformers import (
    Transformation,
    ColumnExpression,
    ColumnSelector,
    columns
)
from skippa.transformers.sklearn import (
    SkippaSimpleImputer,
    SkippaStandardScaler,
    SkippaMinMaxScaler,
    SkippaOneHotEncoder,
    make_skippa_column_transformer
)
from skippa.transformers.custom import (
    SkippaCaster,
    SkippaRenamer,
    SkippaSelector,
    SkippaAssigner,
    SkippaDateEncoder,
    SkippaApplier,
    SkippaConcat
)


PathType = Union[Path, str]


class SkippaPipeline(Pipeline):

    def save(self, file_path: PathType) -> None:
        with open(Path(file_path).as_posix(), 'wb') as f:
            f.write(dill.dumps(self))

    def get_model(self):
        return self.steps[-1][1]


class Skippa:
    """Skippa pipeline class

    A Skippa pipeline can be extended by piping transformation commands.
    Only a number of implemented transformations is supported. 
    Although these transformations use existing scikit-learn transformations, each one reqwuires a 
    specific wrapper  that implements the pandas dataframe support
    """

    def __init__(self, **kwargs) -> None:
        """Create a new Skippa.

        Arguments passed here will be used as arguments for the sklearn Pipeline
        """
        self.pipeline_steps = []
        self._step_idx: int = 0
        self._pipeline_kwargs = kwargs

    def build(self, **kwargs) -> SkippaPipeline:
        """Build into a scikit-learn Pipeline

        Returns:
            Pipeline: An sklearn Pipeline that supports .fit, .transform
        """
        return SkippaPipeline(steps=self.pipeline_steps, **kwargs)

    def _step(self, name: str, transformer: Transformation) -> None:
        """Add a transformation step to the pipeline

        Args:
            name (str): just a descriptive text
            transformer (Transformation): A Skippa-extension of an sklearn transformer
        """
        name = f'{name}_{self._step_idx}'
        self._step_idx += 1
        self.pipeline_steps.append((name, transformer))

    @staticmethod
    def load_pipeline(path: PathType) -> SkippaPipeline:
        """Load a previously saved pipeline

        N.B. dill is used for (de)serialization, because joblib/pickle 
        doesn't support things like lambda functions.

        Args:
            path (PathLike): pathname, either string or pathlib.Path

        Returns:
            SkippaPipeline: an extended sklearn Pipeline
        """
        with open(Path(path).as_posix(), 'rb') as f:
            pipe = dill.loads(f.read())
            if isinstance(pipe, Skippa):
                raise TypeError(
                    "You're using the .load_pipeline method for a Skippa."
                    "Use .load for a saved Skippa"
                    "Use .load_pipeline for a saved Pipeline"
                )
            if not isinstance(pipe, SkippaPipeline):
                raise TypeError(f'This object is not a Skippa, but a {type(pipe)}')
            return pipe

    @staticmethod
    def load(path: PathType) -> Skippa:
        """Load a previously saved skippa

        N.B. dill is used for (de)serialization, because joblib/pickle 
        doesn't support things like lambda functions.

        Args:
            path (PathLike): pathamae, either string or pathlib.Path

        Returns:
            Pipeline: an sklearn Pipeline
        """
        with open(Path(path).as_posix(), 'rb') as f:
            pipe = dill.loads(f.read())
            if isinstance(pipe, SkippaPipeline):
                raise TypeError(
                    "You're using the .load method for a Pipeline."
                    "Use .load for a saved Skippa"
                    "Use .load_pipeline for a saved Pipeline"
                )
            if not isinstance(pipe, Skippa):
                raise TypeError(f'This object is not a Skippa, but a {type(pipe)}')
            return pipe

    def save(self, file_path: PathType) -> None:
        """Save to disk using dill"""
        with open(Path(file_path).as_posix(), 'wb') as f:
            f.write(dill.dumps(self))

    def cast(self, cols: ColumnSelector, dtype: Any) -> Skippa:
        """Cast column to another data type.

        Args:
            cols (ColumnSelector): [description]
            **kwargs: arguments for the actual transformer

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        self._step('cast', SkippaCaster(cols=cols, dtype=dtype))
        return self

    def astype(self, *args, **kwargs) -> Skippa:
        """Alias for .cast"""
        return self.cast(*args, **kwargs)

    def as_type(self, *args, **kwargs) -> Skippa:
        """Alias for .cast"""
        return self.cast(*args, **kwargs)

    def impute(self, cols: ColumnSelector, **kwargs) -> Skippa:
        """Skippa wrapper around sklearn's SimpleImputer

        Args:
            cols (ColumnSelector): [description]

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        self._step('impute', SkippaSimpleImputer(cols=cols, **kwargs))
        return self

    def scale(self, cols: ColumnSelector, type: str = 'standard', **kwargs) -> Skippa:
        """Skippa wrapper around sklearn's StandardScaler / MinMaxScaler

        Args:
            cols (ColumnSelector): [description]
            type (str, optional): One of ['standard', 'minmax']. Defaults to 'standard'.

        Raises:
            ValueError: if an unknown/unsupported scaler type is passed

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        if type == 'standard':
            transformation = SkippaStandardScaler(cols=cols, **kwargs)
        elif type == 'minmax':
            transformation = SkippaMinMaxScaler(cols=cols, **kwargs)
        else:
            raise ValueError(f'Invalid scaler type "{type}". Choose standard or minmax.')
        self._step(f'scale_{type}', transformation)
        return self

    def encode_date(self, cols: ColumnSelector, **kwargs) -> Skippa:
        """A date cannot be used unless you encode it into features.

        This encoder creates new features out of the year, month, day etc.

        Args:
            cols ([type]): [description]
            **kwargs: optional keywords like <datepart>=True/False,
                      indicating whether to use dt.<datepart> as a new feature

        Returns:
            Skippa: [description]
        """
        self._step('date-encode', SkippaDateEncoder(cols=cols, **kwargs))
        return self

    def onehot(self, cols: ColumnSelector, **kwargs) -> Skippa:
        """Skippa wrapper around sklearn's OneHotEncoder

        Args:
            cols (ColumnSelector): [description]

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        if cols is None:
            cols = columns(dtype_include='category')

        kwargs['sparse'] = False
        self._step('onehot', SkippaOneHotEncoder(cols=cols, **kwargs))
        return self

    def rename(self, *args, **kwargs) -> Skippa:
        """Rename certain columns.

        Two ways to use this:
        - a dict which defines a mapping {existing_col: new_col}
        - a column selector and a renaming function (e.g. ['a', 'b', 'c'], lambda c: f'new_{c}')
        It adds an XRenamer step, which wraps around pandas.rename

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        if len(args) == 2:
            cols_to_rename = columns(args[0])
            renamer = args[1]
            assert isinstance(renamer, Callable), 'new names should be a function'
            mapping = (cols_to_rename, renamer)
        elif len(args) == 1 and isinstance(args[0], dict):
            mapping = args[0]
        else:
            mapping = kwargs
        self._step('rename', SkippaRenamer(mapping=mapping))
        return self

    def select(self, cols: ColumnSelector) -> Skippa:
        """Apply a column selection

        Args:
            cols (ColumnSelector): [description]

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        self._step('select', SkippaSelector(cols))
        return self

    def assign(self, **kwargs) -> Skippa:
        """Create new columns based on data in existing columns

        This is a wrapper around pandas' .assign method and uses the same syntax.

        Arguments:
            **kwargs: keyword args denoting new_column=assignment_function pairs

        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        self._step('assign', SkippaAssigner(**kwargs))
        return self

    def apply(self, *args, **kwargs) -> Skippa:
        """Apply a function to the dataframe.

        This is a wrapper around pandas' .apply method and uses the same syntax.

        Arguments:
            *args: first arg should be the funciton to apply
            **kwargs: e.g. axis to apply function on
            
        Returns:
            Skippa: just return itself again (so we can use piping)
        """
        self._step('apply', SkippaApplier(*args, **kwargs))
        return self

    def model(self, model: BaseEstimator) -> SkippaPipeline:
        """Add a model estimator.

        A model estimator is always the last step in the pipeline.
        Therefore this doesn't return the Skippa object (self)
        but calls the .build method to return the pipeline

        Args:
            model (BaseEstimator): An sklearn estimator

        Returns:
            SkippaPipeline: a built pipeline
        """
        expected = [RegressorMixin, ClassifierMixin, ClusterMixin]
        assert any([isinstance(model, cls) for cls in expected]), 'Model should be an sklearn model'
        self._step('model', model)
        return self.build(**self._pipeline_kwargs)

    def __add__(self, pipe: Skippa) -> Skippa:
        """Append two Skippas.

        Q: So when does this make sense?
        A: If you have defined a standard Skippa with transformations you want to do 
           most of the time (e.g. imputation, scaling, whatever) you can define 
           standard skippas and reuse them by adding them to you custom skippa

        Args:
            pipe: Skippa: [description]

        Returns:
            Skippa: [description]
        """
        self.pipeline_steps.extend(pipe.pipeline_steps)
        return self

    def append(self, pipe: Skippa) -> Skippa:
        """Just an alias for adding"""
        return self.__add__(pipe)

    def concat(self, pipe: Skippa) -> Skippa:
        """Concatenate output of this pipeline to another.

        Where adding/appending extends the pipeline, concat keeps
        parallel pipelines and concatenates their outcomes.

        Args:
            pipe (Skippa): [description]

        Returns:
            Skippa: [description]
        """
        new_pipe = Skippa()
        new_pipe._step(
            'concat', 
            SkippaConcat(
                left=('part1', self.build()),
                right=('part2', pipe.build())
            )
        )
        return new_pipe
