# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

RUN python -m pip install mlflow psycopg2-binary boto3

# Install useful commandline tools
RUN apt-get update \
    && apt-get install -y curl htop vim ncdu tree \
    && rm -rf /var/lib/apt/lists/*

CMD mlflow server --host 0.0.0.0 --port ${MLFLOW_TRACKING_PORT} \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:${POSTGRES_PORT}/${POSTGRES_DB} \
    --artifacts-destination s3://mlflow-artifacts
