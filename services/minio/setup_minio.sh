#!/bin/bash

minio_setup() {
  # Wait for Minio to start
  sleep 10

  # Create the bucket
  mc alias set local http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
  mc mb local/${TIMESERIES_BUCKET_NAME} || true
}

# Run the setup function in the background.
# TODO: This feel's hacky. There's probably a better way.
minio_setup &

# I've overwritten the base image (minio's) ENTRYPOINT with my
# own, so call the minio image entrypoint now.
./usr/bin/docker-entrypoint.sh

# Start minio server
minio server --address 0.0.0.0:9000 --console-address 0.0.0.0:9001 /data
