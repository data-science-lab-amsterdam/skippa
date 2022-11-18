# History

## 0.1.15 (2022-11-18)
- Fix: when saving a pipeline, include dependencies in dill serialization.

## 0.1.14 (2022-05-13)
- Bugfix in .assign: shouldn't have columns
- Bugfix in imputer: explicit missing_values arg leads to issues
- Used space-titanic data in examples
- Logo added :)

## 0.1.13 (2022-04-08)
- Bugfix in imputer: using strategy='constant' threw a TypeError when used on string columns

## 0.1.12 (2022-02-07)
- Gradio & dependencies are not installed by default, but are declared an optional extra in setup

## 0.1.11 (2022-01-13)
- Example added for hyperparameter tuning with Hyperopt

## 0.1.10 (2021-12-28)
- Added support for PCA (including example)
- Gradio app support extended to regression
- Minor cleanup and improvements

## 0.1.9 (2021-12-24)
- Added support for automatic creation of Gradio app for model inspection
- Added example with Gradio app

## 0.1.8 (2021-12-23)
- Removed print statement in SkippaSimpleImputer
- Added unit tests

## 0.1.7 (2021-12-20)
- Fixed issue that GridSearchCV (or hyperparam in general) did not work on Skippa pipeline
- Example added using GridSearch

## 0.1.6 (2021-12-17)
- Docs, setup, readme updates
- Updated `.apply()` method so that is accepts a columns specifier

## 0.1.5 (2021-12-13)
- Fixes for readthedocs

## 0.1.4 (2021-12-13)
- Cleanup/fix in examples/full-pipeline.py

## 0.1.3 (2021-12-10)
- Added `.apply()` transformer for `pandas.DataFrame.apply()` functionality
- Documentation and examples update

## 0.1.2 (2021-11-28)
- Added `.assign()` transformer for `pandas.DataFrame.assign()` functionality
- Added `.cast()` transformer (with aliases `.astype()` & `.as_type()`) for `pandas.DataFrame.astype` functionality

## 0.1.1 (2021-11-22)
- Fixes and documentation.

## 0.1.0 (2021-11-19)
- First release on PyPI.
