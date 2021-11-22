"""
import numpy as np
import pandas as pd
import dill

from skippa import Skippa, columns


df = pd.DataFrame({
    'q': [2, 3, 4],
    'x': ['a', 'b', 'c'],
    'y': [1, 16, 1000],
    'z': [0.4, None, 8.7]
})
y = np.array([0, 0, 1])

pipe = (
    Skippa()
        .impute(columns(dtype_include='number'), strategy='median')
        .scale(columns(dtype_include='number'), type='standard')
        .onehot(columns(['x']))
        .rename(columns(pattern='x_*'), lambda c: c.replace('x', 'cat'))
        .select(columns(['y', 'z']) + columns(pattern='cat_*'))
        .build(verbose=True)
)

model = pipe.fit(X=df, y=y)
res = model.transform(df)

"""
from __future__ import annotations
from os import PathLike

from typing import Optional, Union, List, Callable, Type
from pathlib import Path

import dill
import pandas as pd
from sklearn.pipeline import Pipeline

from skippa.transformers import (
    Transformation,
    ColumnExpression,
    XRenamer,
    XSelector,
    XSimpleImputer,
    XStandardScaler,
    XMinMaxScaler,
    XOneHotEncoder,
    XConcat,
    xmake_column_transformer,
    ColumnSelector,
    columns
)


PathLike = Union[Path, str]


class Skippa:
    """Skippa pipeline class

    A Skippa pipeline can be extended by piping transformation commands.
    Only a number of implemented transformations is supported. 
    Although these transformations use existing scikit-learn transformations, each one reqwuires a 
    specific wrapper  that implements the pandas dataframe support
    """

    def __init__(self) -> None:
        """Create a new Skippa. No parameters needed."""
        self.pipeline_steps = []
        self._step_idx: int = 0

    def build(self, **kwargs) -> Pipeline:
        """Build into a scikit-learn Pipeline

        Returns:
            Pipeline: An sklearn Pipeline that supports .fit, .transform
        """
        return Pipeline(steps=self.pipeline_steps, **kwargs)

    def _step(self, name: str, transformer: Transformation) -> None:
        """Add a transformation step to the pipeline

        Args:
            name (str): just a descriptive text
            transformer (Transformation): A Skippa-extension of an sklearn transformer
        """
        name = f'{name}_{self._step_idx}'
        self._step_idx += 1
        self.pipeline_steps.append((name, transformer))

    # def _transformer(self, cols, cls: Type[Transformation], **kwargs) -> Transformation:
    #     try:
    #         custom_class_name = eval(f'X{cls.__name__}')
    #         return custom_class_name(cols=cols, **kwargs)
    #     except NameError:
    #         raise NotImplementedError(f'No custom class implemented for {cls} transformer')

    @staticmethod
    def load_pipeline(path: PathLike) -> Pipeline:
        """Load a previously saved pipeline

        N.B. dill is used for (de)serialization, because joblib/pickle 
        doesn't support things like lambda functions.

        Args:
            path (PathLike): pathamae, either string or pathlib.Path

        Returns:
            Pipeline: an sklearn Pipeline
        """
        with open(Path(path).as_posix(), 'rb') as f:
            pipe = dill.loads(f.read())
            return pipe

    def impute(self, cols: ColumnSelector, **kwargs) -> Skippa:
        """Skippa wrapper around sklearn's SimpleImputer

        Args:
            cols (ColumnSelector): [description]

        Returns:
            Skippa: [description]
        """
        self._step('impute', XSimpleImputer(cols=cols, **kwargs))
        return self

    def scale(self, cols: ColumnSelector, type: str = 'standard', **kwargs) -> Skippa:
        """Skippa wrapper around sklearn's StandardScaler / MinMaxScaler

        Args:
            cols (ColumnSelector): [description]
            type (str, optional): One of ['standard', 'minmax']. Defaults to 'standard'.

        Raises:
            ValueError: if an unknown/unsupported scaler type is passed

        Returns:
            Skippa: [description]
        """
        if type == 'standard':
            transformation = XStandardScaler(cols=cols, **kwargs)
        elif type == 'minmax':
            transformation = XMinMaxScaler(cols=cols, **kwargs)
        else:
            raise ValueError(f'Invalid scaler type "{type}". Choose standard or minmax.')
        self._step(f'scale_{type}', transformation)
        return self

    def onehot(self, cols: ColumnSelector, **kwargs) -> Skippa:
        """Skippa wrapper around sklearn's OneHotEncoder

        Args:
            cols (ColumnSelector): [description]

        Returns:
            Skippa: [description]
        """
        if cols is None:
            cols = columns(dtype_include='category')

        kwargs['sparse'] = False
        self._step('onehot', XOneHotEncoder(cols=cols, **kwargs))
        return self

    def rename(self, *args, **kwargs) -> Skippa:
        if len(args) == 2:
            cols_to_rename = columns(args[0])
            renamer = args[1]
            assert isinstance(renamer, Callable), 'new names should be a function'
            mapping = (cols_to_rename, renamer)
        elif len(args) == 1 and isinstance(args[0], dict):
            mapping = args[0]
        else:
            mapping = kwargs
        self._step('rename', XRenamer(mapping=mapping))
        return self

    def select(self, cols: ColumnSelector) -> Skippa:
        self._step('select', XSelector(cols))
        return self

    def __add__(self, pipe: Skippa) -> Skippa:
        """Append two Skippas -> does this make sense????

        Args:
            pipe: Skippa: [description]

        Returns:
            Skippa: [description]
        """
        self.pipeline_steps.extend(pipe.pipeline_steps)
        return self

    def concat(self, pipe: Skippa) -> Skippa:
        """Concatenate output of this pipeline to another

        Args:
            pipe (Skippa): [description]

        Returns:
            Skippa: [description]
        """
        new_pipe = Skippa()
        new_pipe._step(
            'concat', 
            XConcat(
                left=('part1', self.build()),
                right=('part2', pipe.build())
            )
        )
        return new_pipe
