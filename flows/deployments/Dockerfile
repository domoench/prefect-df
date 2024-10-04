# syntax=docker/dockerfile:1

FROM 326060909633.dkr.ecr.us-west-2.amazonaws.com/prefect-df/flows/flow-deps:latest

ENV DF_ENVIRONMENT=prod
ENV DVC_GIT_USERNAME=domoench
ENV DVC_GIT_REPONAME=df-dvc-prod
ENV DVC_LOCAL_REPO_PATH=/tmp/df-dvc-prod/
ENV MLFLOW_SERVICE_ENDPOINT_PRIVATE=mlflow.prod.prefect-df.local
ENV MLFLOW_TRACKING_PORT=5001

COPY . /opt/prefect/prefect-df/
