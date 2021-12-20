"""
This contains base / utility classes and functions needed for defining/using transformers
"""
from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple, Callable
import re
import logging
import inspect

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer


class ColumnSelector:
    """This is not a transformer, but a utility class for defining a column set."""

    def __init__(self, selector: Callable) -> None:
        self.selector = selector
        self.name = re.sub('[^a-zA-Z0-9_]', '', f'select_{selector}')

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """A ColumnsSelector can be called on a dataframe.

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
        def _ordered_union(list1, list2):
            intersection = set(list1) & set(list2)
            return list1 + [x for x in list2 if x not in intersection]

        return ColumnSelector(
            lambda df: _ordered_union(self.__call__(df), other(df))
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


class SkippaMixin:
    """Utility class providing additional methods for custom Skippa transformers."""
    
    def _get_param_names(self) -> List[str]:
        """Get parameter names for the estimator.
        
        This overrides sklearn's BaseEstimator._get_param_names()
        These things are changed:
        - the first line doesn't look at cls.__init__, but super().__init__
        - it's no longer a class method but an instance method, because the super().__init__ depends on which class was instantiated
        - we manually add the 'cols' parameter, because it's always added to the skippa extension

        This is a fix for a problem using GridSearchCV (or any other sklearn hyperparam search)
        - When calling .fit on the search, it calls .get_params() for each pipeline step
        - this in turn calls the BaseEstimator._get_param_names() class method
        - since our steps are not the sklearn transformers, but Skippa extensions, they have a different init signature
        - this fix makes sure we look at the param signature of the sklearn class (and we mamnually add the 'cols' parameter)
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(super().__init__, "deprecated_original", super().__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (self, init_signature)
                )
        # Extract and sort argument names excluding 'self']
        return sorted([p.name for p in parameters] + ['cols'])
    
    def _set_columns(self, cols: ColumnSelector) -> None:
        self.cols = cols

    def _evaluate_columns(self, X) -> List[str]:
        self._column_names = self.cols(X)
        if len(self._column_names) == 0:
            logging.warn(f'No columns found for column selector {self.cols}')
        return self._column_names

    def _get_result(self, X, res) -> pd.DataFrame:
        column_names = self._evaluate_columns(X)
        X.loc[:, column_names] = res
        return X

    def _set_names(self, X) -> None:
        self.names = X.columns

    def _get_names(self):
        return self.names
