"""
Gradio app to inspect a fitted model/pipeline
- simulation (provide data input > model output)
- explanation?
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import gradio as gr

if TYPE_CHECKING:
    from skippa.pipeline import SkippaPipeline


class GradioApp:
    """This creates a Gradio app for inspecting yourt model pipeline.

    This class can automatically define parameters inputs and outputs of the app
    by using the DataProfile of the pipeline. This profile contains metadata of
    the data that the pipeline was fitted on (columns, dtypes, value ranges).
    """

    def __init__(self, pipe: SkippaPipeline) -> None:
        self.pipe = pipe
        self.data_profile = pipe.get_data_profile()
    
    def build(self, **kwargs) -> gr.Interface:
        """Build the app based on the internal data profile.

        Arguments:
            **kwargs: kwargs received by Gradio's `Interface()` initialisation
        Raises:
            TypeError: If a dtype is encoutered that cannot be handled

        Returns:
            gr.Interface: Gradio Interface object -> call .launch to start the app
        """
        # define inputs
        inputs = []
        for info in self.data_profile:
            if info['is_numeric']:
                col_input = gr.inputs.Slider(
                    minimum=info['min_value'], 
                    maximum=info['max_value'], 
                    default=info['median_value'],
                    label=info['name']
                )
            elif info['is_string']:
                non_null_values = [v for v in info['values'] if not pd.isna(v)]
                if len(non_null_values) <= 10:
                    col_input = gr.inputs.Radio(non_null_values, label=info['name'])
                else:
                    col_input = gr.inputs.Dropdown(non_null_values, label=info['name'])
            else:
                raise TypeError(f"Column {info['name']} has dtype that cannot be handled: {info['dtype']}")
            inputs.append(col_input)

        # define outputs
        if self.data_profile.is_classification():
            prediction_type = 'classification'
            outputs = gr.outputs.Label(label='Prediction')
        elif self.data_profile.is_regression():
            prediction_type = 'regression'
            outputs = gr.outputs.Textbox(type='number', label='Prediction')
        else:
            raise Exception('The data profile has no info on the labels available. Has the pipeline been ftted correctly?')

        # define inference function
        args = self.data_profile.column_names
        def _inference(*args):
            df = pd.DataFrame(data=[args], columns=self.data_profile.column_names)
            if prediction_type == 'classification':
                pred = self.pipe.predict_proba(df)[0]
                return {'No': pred[0], 'Yes': pred[1]}
            else:
                return self.pipe.predict(df)[0]

        # define default args
        app_args = {
            'title': 'Model simulation app',
            'description': 'Change input values and see the effect on model predictions',
            'article': None,
            'theme': 'default',
            'interpretation': None,
            'live': True,
        }
        # override args supplied by user
        app_args.update(kwargs)

        app = gr.Interface(
            fn=_inference,
            inputs=inputs,
            outputs=outputs,
            **app_args
        )
        return app
