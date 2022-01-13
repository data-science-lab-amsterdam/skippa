
"""
Advanced hyperparameter tuning with Hyperopt (if sklearn's GridSearchCV or RandomSearchCV isn't good enough)

this requires pip install hyperopt

How this works:
- Hyperopt requires you to define a function to minimize, that returns the loss for a given parameter set
- this function needs to fit the pipeline using that parameter set
- we can do this using the .set_params() method from sklearn

N.B. Hyperopt is a bit weird with data types, so we need to ensure 
"""
import time
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from skippa import Skippa, columns
from skippa.utils import get_dummy_data


RANDOM_SEED = 123


# get some data
X, y = get_dummy_data(nrows=500, nfloat=3, nchar=1, nint=0, ndate=0, binary_y=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)

# define the pipeline
pipe = (
    Skippa()
        .impute(columns(dtype_include='number'), strategy='median')
        .impute(columns(dtype_include='object'), strategy='most_frequent')
        .onehot(columns(dtype_include='object'), handle_unknown='ignore')
        .model(RandomForestRegressor())
)

# define parameter search space
param_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 2000, 1)),  # explicitly cast to int, of the RandomForestRegressor will throw a TypeError
    'max_depth' : scope.int(hp.quniform('max_depth', 2, 10, 1)),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])
}
pipe_param_space = pipe.get_pipeline_params(param_space)  # this prepends the pipeline step name to each param name

# define the function to minimize: evaluates single parameter selection
def get_cv_score_for_params(params):
    logging.debug(params)
    # set parameter values in the pipeline
    pipe_copy = deepcopy(pipe)
    pipe_copy.set_params(**params)

    # get cross-validation score for selected parameter values
    cv_folds = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    score = -1 * cross_val_score(pipe_copy, X_train, y_train, cv=cv_folds, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return {
        'loss': score,
        'status': STATUS_OK,
        'params': params,
        'eval_time': time.time()
    }

# trials will contain logging information
trials = Trials()

# apply hyperopt's search using the fmin function
best = fmin(
    fn=get_cv_score_for_params,  # function to minimize
    space=pipe_param_space, 
    algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
    max_evals=50,  # maximum number of iterations
    trials=trials,  # logging
    rstate=np.random.default_rng(RANDOM_SEED) 
)

# This is weird: the datatypes are all messed up
# and the pipeline step prefix had been removed from the parameter names
print(best)


# let's fix this:

def results_to_df(trials):
    """Get all the results in a readable format"""
    data=[
        [float(r['loss'])] + [v for v in r['params'].values()]
        for r in trials.results
    ]
    columns=['loss'] + list(trials.results[0]['params'].keys())
    return pd.DataFrame(data=data, columns=columns).sort_values(by='loss', ascending=True)

res = results_to_df(trials)

# we can get the params of the best model like this:
best_params = dict(res.iloc[0, 1:])

# train a model (pipeline) on the full training set using the best params
pipe.set_params(**best_params)
pipe.fit(X_train, y_train)
