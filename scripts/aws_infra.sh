#!/bin/bash

serviceNames=$(aws ecs list-services --cluster $AWS_ECS_CLUSTER | jq '.serviceArns.[] | split("/") | .[2]')

function describe_infra() {
  echo "  RDS Instance: ${AWS_RDS_INSTANCE_ID}"
  echo "  ECS Cluster: ${AWS_ECS_CLUSTER}"
  echo "  ECS Service Names:"
  for serviceName in $serviceNames;
  do
    echo "    $serviceName"
  done
}

if [ $1 = 'start' ]; then

  echo "Starting AWS Infra"
  describe_infra

  # RDS
  echo "Starting RDS instance: ${AWS_RDS_INSTANCE_ID}"
  aws rds start-db-instance --db-instance-identifier $AWS_RDS_INSTANCE_ID > /dev/null

  # EC2 Jumpbox
  # TODO

  # ECS

  # TODO debug why the loop doesn't work for this
  # echo "Updating desired ECS service task count to 1"
  # for service in $serviceNames;
  # do
  #   echo "Updating service $service"
  #   aws ecs update-service \
  #     --cluster $AWS_ECS_CLUSTER \
  #     --service $service \
  #     --desired-count 1
  # done

  echo "Updating ECS services' desired task counts to 1"
  aws ecs update-service \
    --cluster $AWS_ECS_CLUSTER \
    --service prefect-df-prod-ecs-worker-Service-MM3VUmy2n8NU \
    --desired-count 1 \
    > /dev/null

  aws ecs update-service \
    --cluster $AWS_ECS_CLUSTER \
    --service \prefect-df-prod-mlflow-Service-J4WXZZgxgZ8D \
    --desired-count 1 \
    > /dev/null
fi

if [ $1 = 'stop' ]; then
  echo "Stopping AWS Infra"
  describe_infra

  # RDS
  echo "Stopping RDS instance: ${AWS_RDS_INSTANCE_ID}"
  aws rds stop-db-instance --db-instance-identifier $AWS_RDS_INSTANCE_ID > /dev/null

  # EC2 Jumpbox
  # TODO

  # ECS
  echo "Updating ECS services' desired task counts to 0"
  aws ecs update-service \
    --cluster $AWS_ECS_CLUSTER \
    --service prefect-df-prod-ecs-worker-Service-MM3VUmy2n8NU \
    --desired-count 0 \
    > /dev/null

  aws ecs update-service \
    --cluster $AWS_ECS_CLUSTER \
    --service \prefect-df-prod-mlflow-Service-J4WXZZgxgZ8D \
    --desired-count 0 \
    > /dev/null
fi

echo "Done"
