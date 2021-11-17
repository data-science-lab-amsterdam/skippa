"""
Test
"""
import pandas as pd
import dill

from skippa.pipeline import columns, Skippa


df = pd.DataFrame({
    'x': ['a', 'b', 'c'],
    'y': [1, 16, 1000],
    'z': [0.4, None, 8.7]
})
pipe = (
    Skippa()
        .impute(columns(dtype_include='number'), strategy='median')
        .scale(columns(dtype_include='number'), type='standard')
        .onehot(columns(['x']))
        .rename(columns(pattern='x_*'), lambda c: f'QQ{c}')
        .select(columns(['y', 'z', 'QQx_a']))
        .build(verbose=True)
)

res = pipe.fit_transform(df)
print(res)

# filename = './mypipeline.dill'

# with open(filename, 'wb') as f:
#     f.write(dill.dumps(pipe))

# # with open('./mypipeline.joblib', 'rb') as f:
# #     pipe2 = dill.loads(f.read())


# pipe2 = SklearnPreprocessor.load_pipeline(filename)


# res = pipe2.fit_transform(df)
# print(res)

