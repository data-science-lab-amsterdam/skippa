Introduction
============

SciKIt-learn Pipeline in PAndas

Want to create a machine learning model using pandas & scikit-learn? This should make your life easier.

Skippa helps you to easily create a pre-processing and modeling pipeline, based on scikit-learn transformers but preserving pandas dataframe format throughout all pre-processing. This makes it a lot easier to define a series of subsequent transformation steps, while referring to columns in your intermediate dataframe.

Installation
************

.. code-block:: bash

   $ pip install skippa


Basic use
*********
Skippa helps you to easily define data cleaning & pre-processing operations on a pandas DataFrame and combine it with a scikit-learn model/algorithm into a single executable pipeline. It works roughly like this:

.. code-block:: python
   
   from skippa import Skippa, columns
   from sklearn.linear_model import LogisticRegression
   pipeline = (
      Skippa()
      .impute(columns(dtype_include='object'), strategy='most_frequent')
      .impute(columns(dtype_include='number'), strategy='median')
      .scale(columns(dtype_include='number'), type='standard')
      .onehot(columns(['category1', 'category2']))
      .model(LogisticRegression())
   )
   pipeline.fit(X, y)
   predictions = pipeline.predict_proba(new_data)
