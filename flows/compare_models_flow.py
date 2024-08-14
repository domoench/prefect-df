from prefect import flow, task
from mlflow import MlflowClient
from pprint import pprint
from typing import List
from core.types import MLFlowModelSpecifier
import os
import mlflow

"""TODO
    [ ] Parallel fetch and eval of models?
    [ ] Determine which performance metric to use
    [ ] Fetch evaluation test set from EIA
        [ ] Apply same data cleaning as used in training
    [ ] Great expectations test that dataset timespan used to train
        model correctly leaves room for evaluation window.
"""


@task
def evaluate_model(model: mlflow.pyfunc.PyFuncModel):
    def predict(X):
        return model.predict(X)

    eval_data = [] # TODO fetch from EIA

    with mlflow.start_run():
        # Evaluate the function without logging the model
        result = mlflow.evaluate(
            predict,
            eval_data,
            targets='D',
            model_type='regressor',
            evaluators=['default'],  # TODO what is this?
        )
        print(f'metrics:\n{result.metrics}')
        print(f'artifacts:\n{result.artifacts}')


@flow(log_prints=True)
def compare_models(model_specifiers: List[MLFlowModelSpecifier]):

    mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_ENDPOINT_URI'))

    # Fetch models from registry
    models = []
    for ms in model_specifiers:
        print(f'Fetching model from mlflow registry: {ms}')
        model_uri = f'models:/{ms.name}/{ms.version}'
        model = mlflow.pyfunc.load_model(model_uri)
        pprint(model)
        models.append(model)

    for model in models:
        # Evaluate the model
        evaluate_model(model)

        # TODO Emit metrics to MLFlow

    # Fetch performance metrics
    # Generate performance-over-time plot
    # Log plot as artifact to MLFlow
