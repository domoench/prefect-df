from prefect import flow, task
from utils.storage import get_s3_client


@task
def extract():
    print("Downloading timeseries.")
    # TODO: Fetch EIA data
    # TODO: Demonstrate concurrent fetch of paginated API requests


@task
def transform():
    print("Transforming timeseries.")
    # Save into a dataframe
    # Convert types
    # Imputation? Or save that for later down the pipe?


@task
def load():
    print("Loading timeseries into warehouse.")
    # TODO: Convert dataframe to parquet file
    # TODO: Encode start and end times into file name
    # TODO: Write to S3
    s3_client = get_s3_client()
    buckets = s3_client.list_buckets()
    print(buckets)


@flow(log_prints=True)
def etl():
    """Pulls all available hourly EIA demand data up to the last XXXday and
    persists it in the S3 data warehouse as a parquet file.
    """
    extract()
    transform()
    load()
