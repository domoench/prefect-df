#!/bin/bash

echo "Building and pushing python dependencies base image."
docker compose build flow-deps
docker push ${AWS_ECR_ENDPOINT}/prefect-df/flows/flow-deps:latest

# Build flow execution images and push to ECR
flowNames=('etl_flow' 'train_model_flow' 'etl_and_train_flow')
for flowName in "${flowNames[@]}"; do
  echo "Deploying flow: $flowName"
  prefect deploy --name $flowName --prefect-file "flows/deployments/${flowName}.yaml"
done
