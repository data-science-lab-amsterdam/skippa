"""
Example of a full pipeline that also use GridSearch for hyperparameter tuning

This shows:
- how to define the pipeline
- how GridSeaerch works with a Skippa pipeline
- how to use train / test set
- have to save a model/pipeline and how to load/reuse it
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from skippa import Skippa, columns


def get_dummy_data(num_rows: int = 100):
    columns = [chr(i) for i in range(97, 107)]
    data = pd.DataFrame(columns=columns)
    idx = 0
    # 4 floats
    n = 4
    data[columns[idx:idx+n]] = np.random.random((num_rows, n)) * np.array([-1, 1, 10, 100])
    idx += n
    # 2 ints
    n = 2
    data[columns[idx:idx+n]] = np.random.randint(10, size=(num_rows, n))
    idx += n
    # 3 chars
    n = 3
    start = 97
    for i in range(n):
        values = [chr(i) for i in np.random.randint(5, size=num_rows) + start]
        data.iloc[:, idx] = values
        start += 5
        idx += 1
    # 1 date
    dates = [datetime.now() + timedelta(days=-i) for i in range(num_rows)]
    data.iloc[:, idx] = [f'{d:%Y-%m-%d}' for d in dates]

    # set missing values
    data.iloc[3, 0] = np.nan
    data.iloc[4, 1] = np.nan
    data.iloc[5, 7] = None

    y = (np.random.random((num_rows,)) >= 0.75) * 1.
    return data, y


def main():
    # get some data
    X, y = get_dummy_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # define the pipeline
    pipe = (
        Skippa()
            .impute(columns(dtype_include='number'), strategy='median')
            .impute(columns(dtype_include='object'), strategy='most_frequent')
            .scale(columns(dtype_include='number'), type='standard')
            .select(columns(exclude=['a', 'f', 'i', 'j']))
            .onehot(columns(['g', 'h']), handle_unknown='ignore')
            .model(LogisticRegression())
    )

    step_names = list(pipe.named_steps.keys())
    params = {
        # 'impute_0__strategy': ['median'],
        # 'impute_1__strategy': ['most_frequent'],
        'model_5__C': [0.1, 1., 10.]
    }
    search = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=3, 
        error_score='raise',
        verbose=3
    )

    # fit the pipeline on the data
    search.fit(X=X_train, y=y_train)

    print(f"Best parameters (CV score={search.best_score_:.2f}):")
    print(search.best_params_)

    # get model info
    print('Best model coefficients:')
    best_pipe = search.best_estimator_
    best_model = best_pipe.get_model()
    print(best_model.intercept_, best_model.coef_)
    
    # Save the best pipeline to disk
    filename = './mypipeline.dill'
    best_pipe.save(filename)

    # evaluate the model on the test set
    # calling .predict_proba() on the search object or on the best pipeline should lead to the same result
    y_pred = search.predict_proba(X_test)[:, 1]
    y_pred2 = best_pipe.predict_proba(X_test)[:, 1]
    assert all(y_pred2 == y_pred)

    auroc = roc_auc_score(y_true=y_test, y_score=y_pred)
    print(f'AUROC on test set: {auroc:.2f}')


if __name__ == '__main__':
    main()
