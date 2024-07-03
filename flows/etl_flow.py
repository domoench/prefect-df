from prefect import flow
from prefect.deployments import Deployment

@flow(log_prints=True)
def etl():
    print("Downloading timeseries.")
    print("Transforming timeseries.")
    print("Loading timeseries into warehouse.")

if __name__ == "__main__":

# I'm attempting to deploy my flow and have it run on a local worker with
# the flow stored in that local workers file system.

# The commented out approach below doesn't work, the worker successfully
# notices the deployment run, but errors with:
# FileNotFoundError: [Errno 2] No such file or directory: '/opt/prefect/flows/etl_flow.py'
# Even though I confirmed that file does indeed exist.
#    etl.deploy(
#        name="etl",
#        work_pool_name="lf-dev",
#        image="docker.io/prefecthq/prefect:2-python3.10",
#        build=False,
#    )

# I thought perhaps it was because the path/endpoint was not properly defined somehow.
# So I tried the more manual approach below, but no change.
    deployment = Deployment.build_from_flow(
        flow=etl,
        name="etl",
        work_pool_name="lf-dev",
        path="/opt/prefect/",
        entrypoint="flows/etl_flow.py:etl"
    )
    deployment.apply()
