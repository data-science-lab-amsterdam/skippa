# Skippa

SciKIt-learn Pipeline in PAndas

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



## To Do
[ ] Validation of pipeline steps
[ ] Input validation in transformers
[ ] Support arbitrary transformer (if column-preserving)
[ ] Investigate if Skippa can directly extend sklearn's Pipeline


## Credits

This package was created with _Cookiecutter_ and the _`audreyr/cookiecutter-pypackage`_ project template.

- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
