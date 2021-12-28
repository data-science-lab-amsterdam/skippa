"""
Example of applying PCA in Skippa
see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
from sklearn.linear_model import LogisticRegression

from skippa import Skippa, columns
from skippa.utils import get_dummy_data


# get some data
X, y = get_dummy_data(nrows=200, nfloat=20, nint=0, nchar=0, ndate=0, missing=False, binary_y=True)

# define model pipeline
pipe = (
    Skippa()
    .scale(columns())
    .pca(columns(), n_components=3)
    .model(LogisticRegression())
)
pipe.fit(X, y)

# you can access information on the PCA like this:
pipe.named_steps['pca_1'].explained_variance_ratio_
