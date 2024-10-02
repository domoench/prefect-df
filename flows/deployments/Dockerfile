# syntax=docker/dockerfile:1

ARG AWS_ECR_ENDPOINT
FROM $AWS_ECR_ENDPOINT/prefect-df/flows/flow-deps:latest

COPY . /opt/prefect/prefect-df/
