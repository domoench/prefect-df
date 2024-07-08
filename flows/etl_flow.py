from prefect import flow, task
from utils.storage import get_s3_client


@flow(log_prints=True)
def etl():
    print("Downloading timeseries.")
    print("Transforming timeseries.")
    load()


@task
def load():
    print("Loading timeseries into warehouse.")
    s3_client = get_s3_client()
    buckets = s3_client.list_buckets()
    print(buckets)
