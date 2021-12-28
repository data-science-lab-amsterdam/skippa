"""
Gradio app to inspect a fitted model/pipeline
See https://gradio.app

What can this app do?
- simulation (provide data input > model output)
- explanation: this doesn't work well yet. shap isn't supported for most input types


N.B. Run this script in interactive mode (not from the command-line), otherwise the app
will shut down when the script is finished!
"""
from sklearn.linear_model import LogisticRegression

from skippa import Skippa, columns
from skippa.utils import get_dummy_data


# get some data
X, y = get_dummy_data(nrows=200, nfloat=3, nint=0, nchar=1, ndate=0, binary_y=True)

# define model pipeline
pipe = (
    Skippa()
    .impute(columns(dtype_include='number'), strategy='median')
    .impute(columns(dtype_include='object'), strategy='most_frequent')
    .scale(cols=columns(dtype_include='number'))
    .onehot(cols=columns(dtype_include='object'))
    .model(LogisticRegression())
)
pipe.fit(X, y)

# get and launch the app for inspecting your model pipeline!

# this accepts any parameters from gr.Interface() 
# see https://gradio.app/docs/#interface
app = pipe.create_gradio_app(title="My model inspection app")

# this accepts any parameters from gradio.Interface().launch()
# see https://gradio.app/docs/#launch
app.launch(inbrowser=True, server_port=7777)
