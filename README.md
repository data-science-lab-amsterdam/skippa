![pypi](https://img.shields.io/pypi/v/skippa)
![python versions](https://img.shields.io/pypi/pyversions/skippa)
![downloads](https://img.shields.io/pypi/dm/skippa)
![Build status](https://img.shields.io/azure-devops/build/data-science-lab/Intern/263)

<br><br>
<img src="skippa-logo-transparent.png" alt="logo" width="200"/>

# Skippa 

SciKIt-learn Pre-processing Pipeline in PAndas

> __*Read more in the [introduction blog on towardsdatascience](https://towardsdatascience.com/introducing-skippa-bab260acf6a7)*__



Want to create a machine learning model using pandas & scikit-learn? This should make your life easier.

Skippa helps you to easily create a pre-processing and modeling pipeline, based on scikit-learn transformers but preserving pandas dataframe format throughout all pre-processing. This makes it a lot easier to define a series of subsequent transformation steps, while referring to columns in your intermediate dataframe.

So basically the same idea as `scikit-pandas`, but a different (and hopefully better) way to achieve it.

- [pypi](https://pypi.org/project/skippa/)
- [Documentation](https://skippa.readthedocs.io/)

## Installation
```
pip install skippa
```
Optional, if you want to use the [gradio app functionality](./examples/04-gradio-app.py):
```
pip install skippa[gradio]
```

## Basic usage

Import `Skippa` class and `columns` helper function
```
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from skippa import Skippa, columns
```

Get some data
```
df = pd.DataFrame({
    'q': [0, 0, 0],
    'date': ['2021-11-29', '2021-12-01', '2021-12-03'],
    'x': ['a', 'b', 'c'],
    'x2': ['m', 'n', 'm'],
    'y': [1, 16, 1000],
    'z': [0.4, None, 8.7]
})
y = np.array([0, 0, 1])
```

Define your pipeline:
```
pipe = (
    Skippa()
        .select(columns(['x', 'x2', 'y', 'z']))
        .cast(columns(['x', 'x2']), 'category')
        .impute(columns(dtype_include='number'), strategy='median')
        .impute(columns(dtype_include='category'), strategy='most_frequent')
        .scale(columns(dtype_include='number'), type='standard')
        .onehot(columns(['x', 'x2']))
        .model(LogisticRegression())
)
```

and use it for fitting / predicting like this:
```
pipe.fit(X=df, y=y)

predictions = pipe.predict_proba(df)
```

If you want details on your model, use:
```
model = pipe.get_model()
print(model.coef_)
print(model.intercept_)
```

## (de)serialization
And of course you can save and load your model pipelines (for deployment).
N.B. [`dill`](https://pypi.org/project/dill/) is used for ser/de because joblib and pickle don't provide enough support.
```
pipe.save('./models/my_skippa_model_pipeline.dill')

...

my_pipeline = Skippa.load_pipeline('./models/my_skippa_model_pipeline.dill')
predictions = my_pipeline.predict(df_new_data)
```

See the [./examples](./examples) directory for more examples:
- [01-standard-pipeline.py](./examples/01-standard-pipeline.py)
- [02-preprocessing-only.py](./examples/02-preprocessing-only.py)
- [03a-gridsearch.py](./examples/03a-gridsearch.py)
- [03b-hyperopt.py](./examples/03b-hyperopt.py)
- [04-gradio-app.py](./examples/04-gradio-app.py)
- [05-PCA.py](./examples/05-PCA.py)

## To Do
- [x] Support pandas assign for creating new columns based on existing columns
- [x] Support cast / astype transformer
- [x] Support for .apply transformer: wrapper around `pandas.DataFrame.apply`
- [x] Check how GridSearch (or other param search) works with Skippa
- [x] Add a method to inspect a fitted pipeline/model by creating a Gradio app defining raw features input and model output
- [x] Support PCA transformer
- [ ] Facilitate random seed in Skippa object that is dispatched to all downstream operations
- [ ] fit-transform does lazy evaluation > cast to category and then selecting category columns doesn't work > each fit/transform should work on the expected output state of the previous transformer, rather than on the original dataframe
- [ ] Investigate if Skippa can directly extend sklearn's Pipeline -> using __getitem__ trick
- [ ] Use sklearn's new dataframe output setting
- [ ] Validation of pipeline steps
- [ ] Input validation in transformers
- [ ] Transformer for replacing values (pandas .replace)
- [ ] Support arbitrary transformer (if column-preserving)
- [ ] Eliminate the need to call columns explicitly


## Credits
- Skippa is powered by [Data Science Lab Amsterdam](https://www.datasciencelab.nl)
- This project structure is based on the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
