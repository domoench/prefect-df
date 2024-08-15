from prefect import flow, task, runtime
from mlflow import MlflowClient
from pprint import pprint
from typing import List
from etl_flow import get_eia_data_as_df, transform
from train_model_flow import clean_data, features
from core.types import MLFlowModelSpecifier, MLFlowModelInfo
from core.data import TIME_FEATURES, TARGET
from core.consts import (
    EIA_TEST_SET_HOURS,
    EIA_BUFFER_HOURS,
)
from core.utils import mlflow_model_uri, parse_compact_ts_str
import os
import pandas as pd
import mlflow

"""TODO
    [ ] Parallel fetch and eval of models?
    [ ] Determine which performance metric to use
    [ ] Great expectations test that dataset timespan used to train
        model correctly leaves room for evaluation window.
"""


@task
def fetch_eval_dataset() -> pd.DataFrame:
    # Fetch EIA data for the eval window
    end_ts = (pd.Timestamp.utcnow().round('h') - pd.Timedelta(hours=EIA_BUFFER_HOURS))
    start_ts = end_ts - pd.Timedelta(hours=EIA_TEST_SET_HOURS)
    df = get_eia_data_as_df(start_ts, end_ts)

    # Transform and feature engineer
    df = transform(df)
    df = clean_data(df)
    df = features(df)
    df = df.drop(columns=['respondent'])  # TODO remove this in ETL?
    print(f'Eval data df:\n{df.head()}')
    return df


@task
def evaluate_model(model_info: MLFlowModelInfo, eval_df: pd.DataFrame):
    # TODO assert the model run's training end date leaves room for evaluation window
    end_ts_str = model_info.run.data.params['dvc.dataset.train.end']
    train_end_ts = parse_compact_ts_str(end_ts_str)
    eval_start_ts = eval_df.index.min()
    assert train_end_ts < eval_start_ts

    mlflow.set_experiment('xgb.df.compare_models')
    run_name = f'{model_info.specifier.name}-v{model_info.specifier.version}_eval'
    model = model_info.model
    with mlflow.start_run(run_name=run_name):
        # Evaluate the function without logging the model
        result = mlflow.evaluate(
            model=model,
            data=eval_df,
            targets=TARGET,
            model_type='regressor',
            evaluators=['default'],
        )
        mlflow.log_params({
            'model_name': model_info.specifier.name,
            'model_version': model_info.specifier.version,
        })
        mlflow.set_tags({
            'prefect_flow_run': runtime.flow_run.name,
        })
        print(f'metrics:\n{result.metrics}')
        print(f'artifacts:\n{result.artifacts}')


@flow(log_prints=True)
def compare_models(model_specifiers: List[MLFlowModelSpecifier]):

    mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_ENDPOINT_URI'))
    client = MlflowClient()

    # Fetch models from model registry
    model_details = []
    for ms in model_specifiers:
        print(f'Fetching model from mlflow registry: {ms}')
        model = mlflow.pyfunc.load_model(mlflow_model_uri(ms))
        pprint(model)

        print(f'Fetching mlflow run info for {model.metadata.run_id}')
        run = client.get_run(model.metadata.run_id)
        model_details.append(MLFlowModelInfo(specifier=ms, model=model, run=run))

    eval_df = fetch_eval_dataset()

    for md in model_details:
        # Evaluate the model
        evaluate_model(md, eval_df)

    # Fetch performance metrics
    # Generate performance-over-time plot
    # Log plot as artifact to MLFlow
