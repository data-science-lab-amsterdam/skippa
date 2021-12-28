"""
Example of a full pipeline that also use GridSearch for hyperparameter tuning

This shows:
- how to define the pipeline
- how GridSeaerch works with a Skippa pipeline
- how to use train / test set
- have to save a model/pipeline and how to load/reuse it
"""
from numpy.core.fromnumeric import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from skippa import Skippa, columns
from skippa.utils import get_dummy_data


def main():
    # get some data
    X, y = get_dummy_data(nrows=500, nfloat=3, nchar=1, nint=0, ndate=0, binary_y=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # define the pipeline
    pipe = (
        Skippa()
            .impute(columns(dtype_include='number'), strategy='median')
            .impute(columns(dtype_include='object'), strategy='most_frequent')
            .onehot(columns(dtype_include='object'), handle_unknown='ignore')
            .model(RandomForestRegressor())
    )

    # Define a parameter grid.
    # N.B. Because we have a Pipeline instead of a single estimator, parameter keys should
    # be in the format <pipelinestepname>__<parametername>.
    # There are 2 ways to do this:

    # 1. You can check the names of the pipeline steps, in order to define params for the right step
    step_names = list(pipe.named_steps.keys())
    print(step_names)
    params = {
        'model_3__n_estimators': [100, 200],
        'model_3__max_depth': [3, 5]
    }

    # 2. You can also do this (it will auto-detect the pipeline step name):
    params = pipe.get_pipeline_params({
        'n_estimators': [100, 200],
        'max_depth': [3, 5]
    })

    # define the grid search
    search = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=3, 
        error_score='raise',
        verbose=3
    )

    # fit on the data
    search.fit(X=X_train, y=y_train)

    print(f"Best parameters (CV score={search.best_score_:.2f}):")
    print(search.best_params_)

    # get model info
    print('Feature importances of best model:')
    best_pipe = search.best_estimator_
    best_model = best_pipe.get_model()
    print(best_model.feature_importances_)
    
    # Save the best pipeline to disk
    filename = './mypipeline.dill'
    best_pipe.save(filename)

    # evaluate the model on the test set
    # calling .predict() on the search object or on the best pipeline should lead to the same result
    y_pred1 = search.predict(X_test)
    y_pred2 = best_pipe.predict(X_test)
    assert all(y_pred2 == y_pred1)

    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred1)
    print(f'MAE on test set: {mae:.2f}')


if __name__ == '__main__':
    main()
