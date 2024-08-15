# syntax=docker/dockerfile:1

FROM minio/minio:latest

COPY setup_minio.sh /usr/local/bin/setup_minio.sh
RUN chmod +x /usr/local/bin/setup_minio.sh

# Setup minio then run the server
ENTRYPOINT ["/usr/local/bin/setup_minio.sh"]
