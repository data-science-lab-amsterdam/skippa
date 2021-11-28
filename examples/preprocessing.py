"""
Test data preprocessing
Result of the pipeline is not a fitted model, but a transformed dataframe
"""
import numpy as np
import pandas as pd

from skippa import Skippa, columns


def main():
    df = pd.DataFrame({
        'q': ['2021-11-29', '2021-12-01', '2021-12-03'],
        'x': ['a', 'b', 'c'],
        'x2': ['m', 'n', 'm'],
        'y': [1, 16, 1000],
        'z': [0.4, None, 8.7]
    })
    
    print(df.info())

    # pipe0 = (
    #     Skippa()
    #     .onehot(columns(['y']))
    #     .select(columns(pattern='y_*'))
    # )

    pipe = (
        Skippa()
            .astype(columns(['x2']), 'category')
            .impute(columns(dtype_include='number'), strategy='median')
            .impute(columns(dtype_include=['category', 'object']), strategy='most_frequent')
            .scale(columns(dtype_include='number'), type='standard')
            .encode_date(columns(['q']))
            .onehot(columns(['x', 'x2']))
            .rename(columns(pattern='x_*'), lambda c: c.replace('x', 'cat'))
            .select(columns(['y', 'z']) + columns(pattern='cat_*'))
            #.select(columns(['y', 'z']) + columns(['cat_a', 'cat_b']))
            # .concat(pipe0)
            .assign(y2 = lambda x: x['y'] * 10)
            .build()
    )

    df_processed = pipe.fit_transform(df)
    print(df_processed)


if __name__ == '__main__':
    main()
