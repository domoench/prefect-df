import os
import mlflow
import pandas as pd
from core.data import get_dvc_remote_repo_url
from core.consts import EIA_TEST_SET_HOURS, DVC_EARLIEST_DATA_HOUR, EIA_BUFFER_HOURS
from core.utils import mlflow_endpoint_uri, InvalidExecutionEnvironmentError
from core.model import get_model_features, get_data_for_model_input
from flows.train_model_flow import (
    mlflow_emit_tags_and_params
)


git_repo_url = get_dvc_remote_repo_url()

# Get training data set
df_env = os.getenv('DF_ENVIRONMENT')
if df_env == 'prod':
    raise NotImplementedError
elif df_env == 'dev':
    start_ts = pd.Timestamp(DVC_EARLIEST_DATA_HOUR) + pd.DateOffset(years=3)
    end_ts = pd.Timestamp.utcnow().round('h') - pd.Timedelta(hours=2*EIA_BUFFER_HOURS)
else:
    raise InvalidExecutionEnvironmentError(df_env)


df = get_data_for_model_input(start_ts, end_ts)

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

# Do a quick dummy prediction to infer a signature
features = get_model_features()  # Only baseline features
X = df[features]
y_pred = baseline_model.predict(None, X)
signature = mlflow.models.infer_signature(X, y_pred)

# Register basline model to mlflow
mlflow.set_tracking_uri(uri=mlflow_endpoint_uri())
mlflow.set_experiment('xgb.df.register_baseline')
with mlflow.start_run():
    mlflow_emit_tags_and_params(df)
    mlflow.pyfunc.log_model(
        artifact_path="baseline_model",
        python_model=baseline_model,
        signature=signature
    )
