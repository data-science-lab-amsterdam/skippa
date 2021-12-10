"""
Test
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from skippa import Skippa, columns


def get_dummy_data():
    len = 20
    columns = [chr(i) for i in range(97, 107)]
    data = pd.DataFrame(columns=columns)
    idx = 0
    # 4 floats
    n = 4
    data[columns[idx:idx+n]] = np.random.random((len, n)) * np.array([-1, 1, 10, 100])
    idx += n
    # 2 ints
    n = 2
    data[columns[idx:idx+n]] = np.random.randint(10, size=(len, n))
    idx += n
    # 3 chars
    n = 3
    start = 97
    for i in range(n):
        values = [chr(i) for i in np.random.randint(5, size=len) + start]
        data.iloc[:, idx] = values
        start += 5
        idx += 1
    # 1 date
    dates = [datetime.now() + timedelta(days=-i) for i in range(len)]
    data.iloc[:, idx] = [f'{d:%Y-%m-%d}' for d in dates]

    # set missing values
    data.iloc[3, 0] = np.nan
    data.iloc[4, 1] = np.nan
    data.iloc[5, 7] = None

    y = (np.random.random((len,)) >= 0.75) * 1.
    return data, y


def main():
    # get some data
    X, y = get_dummy_data()

    # define the pipeline
    pipe = (
        Skippa()
            .impute(columns(dtype_include='number'), strategy='median')
            .impute(columns(dtype_include='object'), strategy='most_frequent')
            .scale(columns(dtype_include='number'), type='standard')
            .select(columns(exclude=['a', 'f', 'i', 'j']))
            .onehot(columns(['g', 'h']))
            .rename({'b': 'bbb'})
            # .build()
            .model(LogisticRegression())
    )

    res = pipe.fit_transform(X)
    print(res)
    # exit(0)

    # fit the pipeline on the data
    pipe.fit(X=X, y=y)

    # get model info
    print('Model coefficients:')
    print(pipe.get_model().coef_, pipe.get_model().intercept_)
    
    # save the fitted pipeline to disk
    filename = './mypipeline.dill'
    pipe.save(filename)

    # ...

    # load the pipeline and apply it to some data
    model_pipeline = Skippa.load_pipeline(filename)
    predictions = model_pipeline.predict_proba(X)
    print('Model predictions:')
    print(predictions)


if __name__ == '__main__':
    main()
