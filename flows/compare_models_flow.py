from prefect import flow, task, runtime
from mlflow import MlflowClient
from pprint import pprint
from typing import List
from flows.etl_flow import get_eia_data_as_df, transform
from flows.train_model_flow import clean_data, features
from core.types import MLFlowModelSpecifier, MLFlowModelInfo
from core.consts import EIA_TEST_SET_HOURS, EIA_BUFFER_HOURS, TARGET
from core.utils import mlflow_model_uri, parse_compact_ts_str
import os
import matplotlib.pyplot as plt
import pandas as pd
import mlflow

"""TODO
    [ ] Parallel fetch and eval of models?
    [ ] Determine which performance metric to use
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
    print(f'Eval data df:\n{df}')
    return df


@task
def evaluate_model(model_info: MLFlowModelInfo, eval_df: pd.DataFrame):
    end_ts_str = model_info.run.data.params['dvc.dataset.train.end']
    train_end_ts = parse_compact_ts_str(end_ts_str)
    eval_start_ts = eval_df.index.min()
    print(f'train_end_ts:{train_end_ts}. eval_start:{eval_start_ts}')
    assert train_end_ts <= eval_start_ts
    # TODO: Replace assertion with more comprehensive great expectations validation?

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


@task
def generate_performance_plot(model_names: List[str]):
    # Fetch experiment runs from mlflow
    mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_ENDPOINT_URI'))
    experiment_name = 'xgb.df.compare_models'
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    print(f'Summary of runs for experiment {experiment_name}:')
    print(runs)

    # Metrics of interest
    metrics = [
        'r2_score', 'root_mean_squared_error', 'mean_squared_error',
        'mean_absolute_percentage_error'
    ]

    # Cast some types in the runs dataframe
    runs['start_time'] = pd.to_datetime(runs['start_time'], utc=True)
    runs['params.model_version'] = pd.to_numeric(runs['params.model_version'])
    for m in metrics:
        runs[f'metrics.{m}'] = pd.to_numeric(runs[f'metrics.{m}'])

    # Generate the plot
    num_cols = 3
    num_rows = (len(metrics) // num_cols) + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    axs = axs.flat
    for model_name in model_names:
        model_runs_df = runs[runs['params.model_name'] == model_name]
        model_versions = model_runs_df['params.model_version'].unique()
        for i, metric in enumerate(metrics):
            for v in model_versions:
                df = model_runs_df[model_runs_df['params.model_version'] == v]
                axs[i].set_title(metric)
                axs[i].plot(df.start_time, df[f'metrics.{metric}'],
                            marker='o', label=f'{model_name}-v{v}')
                axs[i].legend()
                axs[i].tick_params(axis='x', rotation=45)

    # Remove axes for any extra subplots beyond the number of metrics
    for i in range(len(metrics), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.close(fig)

    mlflow.set_experiment('xgb.df.compare_models_plot')
    with mlflow.start_run():
        # MLFlow Artifact
        mlflow.log_figure(fig, 'model_eval_comparison.png')


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

    # TODO paralellize these?
    for md in model_details:
        # Evaluate the model
        evaluate_model(md, eval_df)

    # Generate and log performance-over-time plot
    generate_performance_plot([ms.name for ms in model_specifiers])
