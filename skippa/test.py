"""
Test
"""
import numpy as np
import pandas as pd
import dill

from skippa import Skippa, columns


def main():
    df = pd.DataFrame({
        'q': ['2021-11-29', '2021-12-01', '2021-12-03'],
        'x': ['a', 'b', 'c'],
        'x2': ['m', 'n', 'm'],
        'y': [1, 16, 1000],
        'z': [0.4, None, 8.7]
    })
    y = np.array([0, 0, 1])

    # pipe0 = (
    #     Skippa()
    #     .onehot(columns(['y']))
    #     .select(columns(pattern='y_*'))
    # )

    pipe = (
        Skippa()
            .impute(columns(dtype_include='number'), strategy='median')
            .scale(columns(dtype_include='number'), type='standard')
            .encode_date(columns(['q']))
            .onehot(columns(['x', 'x2']))
            .rename(columns(pattern='x_*'), lambda c: c.replace('x', 'cat'))
            #.select(columns(['y', 'z']) + columns(pattern='cat_*'))
            #.select(columns(['y', 'z']) + columns(['cat_a', 'cat_b']))
            # .concat(pipe0)
            .build(verbose=True)
    )

    model = pipe.fit(X=df, y=y)

    res = model.transform(df)
    #res = pipe.fit_transform(df)
    print(res)

    # filename = './mypipeline.dill'

    # with open(filename, 'wb') as f:
    #     f.write(dill.dumps(pipe))

    # # with open('./mypipeline.joblib', 'rb') as f:
    # #     pipe2 = dill.loads(f.read())


    # pipe2 = SklearnPreprocessor.load_pipeline(filename)


    # res = pipe2.fit_transform(df)
    # print(res)

#
# Use case for .add method:
# - you define a 'standard' skippa that you always want to use
# - you can define it as an object and import it
# - you add your current skippa to it
#

if __name__ == '__main__':
    main()
