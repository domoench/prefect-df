# syntax=docker/dockerfile:1

FROM minio/minio:latest

COPY setup_minio.sh /usr/local/bin/setup_minio.sh
RUN chmod +x /usr/local/bin/setup_minio.sh

# Setup minio then run the server
ENTRYPOINT ["/usr/local/bin/setup_minio.sh"]

# Run minio server
# CMD ["minio", "server", "--address", "0.0.0.0:9000", "--console-address", "0.0.0.0:9001", "/data"]
