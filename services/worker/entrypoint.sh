#!/bin/bash

export PYTHONPATH="/opt/prefect:${PYTHONPATH}"

# Register flow deployments with prefect server
echo "Deploying flows."
python /opt/prefect/flows/deploy_flows_dev.py

# Start the Prefect worker and connect to the work pool
WORK_POOL="lf-dev"
echo "Starting prefect worker in pool $WORK_POOL."
prefect worker start --pool $WORK_POOL
