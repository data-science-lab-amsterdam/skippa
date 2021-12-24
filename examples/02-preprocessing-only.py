"""
Test data preprocessing only
In thisd case, the result of the pipeline is not a fitted model, but a transformed dataframe

N.B. You can also use this approach to inspect the contents of the dataframe after your pre-processing
"""
import numpy as np
import pandas as pd

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
    print(df.info())

    # define the pipeline
    pipe = (
        Skippa()
            .astype(columns(['x2']), 'category')
            .rename({'x2': 'otherx'})
            .impute(columns(dtype_include='number'), strategy='median')
            .impute(columns(dtype_include=['category', 'object']), strategy='most_frequent')
            .scale(columns(dtype_include='number'), type='standard')
            .encode_date(columns(['q']))
            .onehot(columns(['x', 'otherx']))
            .assign(y2 = lambda x: x['y'] * 10)
            .build()
    )

    # call .fit_transform to apply it to the data
    df_processed = pipe.fit_transform(df)
    print(df_processed)


if __name__ == '__main__':
    main()
