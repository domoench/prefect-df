#!/bin/bash

mlflow server \
  --host 0.0.0.0
  --port ${MLFLOW_TRACKING_PORT} \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION}
