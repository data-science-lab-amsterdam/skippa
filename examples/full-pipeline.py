"""
Test
"""
import numpy as np
import pandas as pd
import dill

from sklearn.linear_model import LogisticRegression

from skippa import Skippa, columns


def main():
    # get some data
    df = pd.DataFrame({
        'q': ['2021-11-29', '2021-12-01', '2021-12-03'],
        'x': ['a', 'b', 'c'],
        'x2': ['m', 'n', 'm'],
        'y': [1, 16, 1000],
        'z': [0.4, None, 8.7]
    })
    y = np.array([0, 0, 1])

    # define the pipeline
    pipe = (
        Skippa()
            .cast(columns(['x', 'x2']), 'category')
            .impute(columns(dtype_include='number'), strategy='median')
            .impute(columns(dtype_include='category'), strategy='most_frequent')
            .scale(columns(dtype_include='number'), type='standard')
            .onehot(columns(['x', 'x2']))
            .rename(columns(pattern='x_*'), lambda c: c.replace('x', 'cat'))
            .select(columns(['y', 'z']) + columns(pattern='cat_*'))
            .model(LogisticRegression())
    )

    # fit the pipeline on the data
    pipe.fit(X=df, y=y)

    # get model info
    print('Model coefficients:')
    print(pipe.get_model().coef_, pipe.get_model().intercept_)
    
    # save the fitted pipeline to disk
    filename = './mypipeline.dill'
    pipe.save(filename)

    # ...

    # load the pipeline and apply it to some data
    model_pipeline = Skippa.load_pipeline(filename)
    predictions = model_pipeline.predict_proba(df)
    print('Model predictions:')
    print(predictions)


if __name__ == '__main__':
    main()
