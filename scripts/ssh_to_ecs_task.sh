#!/bin/bash

if [[ "$#" -ne 2 ]]; then
  echo "Expected Usage: ./ssh_to_ecs_task.sh [task_id] [container_name]"
  exit
fi

taskID=$1
containerName=$2

aws ecs execute-command \
  --cluster ${MLFLOW_ECS_CLUSTER} \
  --task $taskID \
  --container $containerName \
  --interactive \
  --command "/bin/sh"
