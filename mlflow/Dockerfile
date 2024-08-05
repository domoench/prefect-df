# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

RUN python -m pip install mlflow

# Install useful commandline tools
RUN apt-get update \
    && apt-get install -y curl htop vim ncdu tree \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5001

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]
