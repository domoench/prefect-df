# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Install useful commandline tools
RUN apt-get update \
    && apt-get install -y curl htop vim ncdu tree git \
    && rm -rf /var/lib/apt/lists/*

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create a non-root user
RUN useradd -ms /bin/bash jupyteruser

# Set the working directory and give appropriate permissions to the new user
WORKDIR /app
RUN chown -R jupyteruser:jupyteruser /app
USER jupyteruser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app"]
