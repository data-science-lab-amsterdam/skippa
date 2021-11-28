![pypi](https://img.shields.io/pypi/v/skippa)
![](https://img.shields.io/pypi/pyversions/skippa)


# Skippa 
:kangaroo: easy hopping

SciKIt-learn Pipeline in PAndas

Want to create a machine learning model using pandas & scikit-learn? This should make your life easier.

Skippa helps you to easily create a pre-processing and modeling pipeline, based on scikit-learn transformers but preserving pandas dataframe format throughout all pre-processing. This makes it a lot easier to define a series of subsequent transformation steps, while referring to columns in your intermediate dataframe.

So basically the same idea as `scikit-pandas`, but a different (and hopefully better) way to achieve it.

## Installation
```
pip install skippa
```

## Basic usage

Import Skippa class and `columns` helper
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
        .impute(columns(dtype_include='number'), strategy='median')
        .impute(columns(dtype_include='category'), strategy='most_frequent')
        .scale(columns(dtype_include='number'), type='standard')
        .encode_date(columns(['date']))
        .onehot(columns(['x', 'x2']))
        .rename(columns(pattern='x_*'), lambda c: c.replace('x', 'prop'))
        .select(columns(['y', 'z']) + columns(pattern='prop*'))
        .model(LogisticRegression())
)
```

and use it for fitting / predicting like this:
```
model_pipeline = pipe.fit(X=df, y=y)

predictions = model_pipeline.predict_proba(df)
```

If you want details on your model, use:
```
model = model_pipeline.get_model()
print(model.coef_)
print(model.intercept_)
```

## (de)serialization
And of course you can save and load your model pipelines (for deployment).
N.B. [`dill`](https://pypi.org/project/dill/) is used for ser/de because joblib and pickle don't provide enough support.
```
model_pipeline.save('./models/my_skippa_model_pipeline.dill')

...

my_pipeline = Skippa.load_pipeline('./models/my_skippa_model_pipeline.dill')
predictions = my_pipeline.predict(df_new_data)
```

## To Do
- [x] Support pandas assign for creating new columns based on existing columns
- [x] Support cast / astype transformer
- [ ] Investigate if Skippa can directly extend sklearn's Pipeline
- [ ] Validation of pipeline steps
- [ ] Input validation in transformers
- [ ] Support arbitrary transformer (if column-preserving)
- [ ] Eliminate the need to call columns explicitly
- [ ] Add more transformations


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
