# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

RUN python -m pip install mlflow psycopg2-binary boto3

# Install useful commandline tools
RUN apt-get update \
    && apt-get install -y curl htop vim ncdu tree procps \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
