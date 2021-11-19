"""
Test
"""
import numpy as np
import pandas as pd
import dill

from skippa.pipeline import Skippa, columns


df = pd.DataFrame({
    'q': [2, 3, 4],
    'x': ['a', 'b', 'c'],
    'y': [1, 16, 1000],
    'z': [0.4, None, 8.7]
})
y = np.array([0, 0, 1])

pipe0 = (
    Skippa()
    .onehot(columns(['y']))
    .select(columns(pattern='y_*'))
)

pipe = (
    Skippa()
        .impute(columns(dtype_include='number'), strategy='median')
        .scale(columns(dtype_include='number'), type='standard')
        .onehot(columns(['x']))
        .rename(columns(pattern='x_*'), lambda c: c.replace('x', 'cat'))
        .select(columns(['y', 'z']) + columns(pattern='cat_*'))
        .concat(pipe0)
        .build(verbose=True)
)

model = pipe.fit(X=df, y=y)

res = pipe.transform(df)
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

