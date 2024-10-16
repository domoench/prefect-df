import os
import mlflow
import pandas as pd
from core.data import get_dvc_remote_repo_url, get_dvc_dataset_as_df
from core.types import DVCDatasetInfo
from core.consts import EIA_TEST_SET_HOURS
from core.utils import mlflow_endpoint_uri
from flows.train_model_flow import (
    clean_data, features, mlflow_emit_tags_and_params
)


git_PAT = os.getenv('DVC_GIT_REPO_PAT')
git_repo_url = get_dvc_remote_repo_url(git_PAT)

# Get training data
path = 'data/eia_d_df_2019-01-01_00_2024-09-30_00.parquet'
rev = 'a2c13b81223c8a9f634e5688a10ccf0e1cbbb73c'
dvc_dataset_info = DVCDatasetInfo(repo=git_repo_url, path=path, rev=rev)
df = get_dvc_dataset_as_df(dvc_dataset_info)

df = clean_data(df)
df = features(df)

# Chop off the last EIA_TEST_SET_HOURS hours, so the training set leaves
# enough of a test set window for adhoc model evaluation.
df = df[:-EIA_TEST_SET_HOURS]

# Calculate average demand by month and hour
demand_by_hour_month = df.groupby(['hour', 'month'])['D'].agg(pd.Series.mean)
demand_by_hour_month


# Define baseline model
class BaselineModel(mlflow.pyfunc.PythonModel):
    def __init__(self, demand_by_hour_month):
        self.dbhm = demand_by_hour_month

    def predict(self, context, model_input_df):
        # Assuming model_input contains a column 'hour' with the hour of the data point
        def demand_for_row(row):
            key_tuple = (row['hour'], row['month'])
            return self.dbhm[key_tuple]
        return model_input_df.apply(demand_for_row, axis=1)


baseline_model = BaselineModel(demand_by_hour_month)

# Register basline model to mlflow
mlflow.set_tracking_uri(uri=mlflow_endpoint_uri())
mlflow.set_experiment('xgb.df.register_baseline')
with mlflow.start_run():
    mlflow_emit_tags_and_params(df, dvc_dataset_info)
    mlflow.pyfunc.log_model(artifact_path="baseline_model", python_model=baseline_model)
