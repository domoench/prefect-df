#!/bin/bash

# Build and push python dependencies base image
docker compose build flow-deps
docker push ${AWS_ECR_ENDPOINT}/prefect-df/flows/flow-deps:latest

# Build flow execution images
prefect deploy --name test_flow --prefect-file flows/deployments/test_flow.yaml
