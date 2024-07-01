#!/bin/bash

echo "Prefect API URL: $PREFECT_API_URL"

# Start the Prefect worker and connect to the work pool
prefect worker start --pool "lf"
