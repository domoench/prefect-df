#!/bin/bash

serviceNames=$(aws ecs list-services --cluster $AWS_ECS_CLUSTER | jq '.serviceArns.[] | split("/") | .[2]')

function describe_infra() {
  echo "  RDS Instance: ${AWS_RDS_INSTANCE_ID}"
  echo "  ECS Cluster: ${AWS_ECS_CLUSTER}"
  echo "  EC2 Jumpbox:"
  echo "    Instance ID: ${AWS_JUMPBOX_INSTANCE_ID}"
  echo "    Public DNS: $(aws ec2 describe-instances --instance-ids $AWS_JUMPBOX_INSTANCE_ID | jq  '.Reservations.[].Instances.[].PublicDnsName')"

  echo "  ECS Service Names:"
  for serviceName in $serviceNames;
  do
    echo "    $serviceName"
  done
}

function start_infra() {
  echo "Starting AWS Infra"
  describe_infra

  # RDS
  echo "Starting RDS instance: ${AWS_RDS_INSTANCE_ID}"
  aws rds start-db-instance --db-instance-identifier $AWS_RDS_INSTANCE_ID > /dev/null

  # EC2 Jumpbox
  echo "Starting EC2 Jumpbox instance: ${AWS_JUMPBOX_INSTANCE_ID}"
  aws ec2 start-instances --instance-ids $AWS_JUMPBOX_INSTANCE_ID > /dev/null

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
}

function stop_infra() {
  echo "Stopping AWS Infra"
  describe_infra

  # RDS
  echo "Stopping RDS instance: ${AWS_RDS_INSTANCE_ID}"
  aws rds stop-db-instance --db-instance-identifier $AWS_RDS_INSTANCE_ID > /dev/null

  # EC2 Jumpbox
  echo "Stopping EC2 instance: ${AWS_JUMPBOX_INSTANCE_ID}"
  aws ec2 stop-instances --instance-ids $AWS_JUMPBOX_INSTANCE_ID > /dev/null

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
}

if [[ $1 == 'start' ]]; then
  start_infra
elif [[ $1 = 'stop' ]]; then
  stop_infra
elif [[ $1 = 'describe' ]]; then
  describe_infra
else
  echo "Argument must be 'start', 'stop', or 'describe'"
  exit
fi
echo "Done"
