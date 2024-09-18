#!/bin/bash

aws ecs execute-command \
  --cluster ${MLFLOW_ECS_CLUSTER} \
  --task ${MLFLOW_ECS_TASK_ID} \
  --container ${MLFLOW_ECS_CONTAINER_NAME} \
  --interactive \
  --command "/bin/sh"
