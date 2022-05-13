"""
Example of a full pipeline, i.e. both pre-processing and a modeling algorithm
This shows:
- how to define the pipeline
- how to use train / test set
- have to save a model/pipeline and how to load/reuse it
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skippa import Skippa, columns


def main():
    """
    Example using the Spaceship Titanic dataset (see https://www.kaggle.com/competitions/spaceship-titanic)
    """
    # Read data, define X/y, split train test 
    data = pd.read_csv('examples/space-titanic.csv')
    X = data.drop(['Transported'], axis=1)
    y = data['Transported'].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # Define the pipeline
    pipe = (
        Skippa()
        # convert booleans to float
        .astype(columns(['CryoSleep', 'VIP']), float)
        # the Cabin column contains info on which deck and if it's port or starboard side
        .assign(
            cabin_deck=lambda df: df['Cabin'].str.split('/').str.get(0),
            cabin_portside=lambda df: (df['Cabin'].str.split('/').str.get(2) == 'S').astype(float)
        )
        # remove (deselect) these columns
        .select(columns(exclude=['PassengerId', 'Cabin', 'Name']))
        # impute missing values: median value for numeric cols, most freequent for categorical cols
        .impute(columns(dtype_include='number'), strategy='median')
        .impute(columns(dtype_include='object'), strategy='most_frequent')
        # scale numeric cols
        .scale(columns(dtype_include='number'), type='standard')
        # one-hot-encode categorical cols
        .onehot(columns(dtype_include='object'), handle_unknown='ignore')
        # now the data is ready to fit a classifier
        .model(LogisticRegression())
    )

    # fit the pipeline on the data
    pipe.fit(X=X_train, y=y_train)

    # get model info
    print('Model coefficients:')
    print(pipe.get_model().coef_, pipe.get_model().intercept_)
    
    # save the fitted pipeline to disk
    filename = './mypipeline.dill'
    pipe.save(filename)

    # evaluate the model
    y_pred = pipe.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_true=y_test, y_score=y_pred)
    print(f'AUROC: {auroc:.2f}')

    # ...

    # load the pipeline and apply it to some data
    model_pipeline = Skippa.load_pipeline(filename)
    predictions = model_pipeline.predict_proba(X)
    print('Model predictions:')
    print(predictions)


if __name__ == '__main__':
    main()
