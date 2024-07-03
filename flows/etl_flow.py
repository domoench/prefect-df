from prefect import flow
from prefect.deployments import Deployment

@flow(log_prints=True)
def etl():
    print("Downloading timeseries.")
    print("Transforming timeseries.")
    print("Loading timeseries into warehouse.")

if __name__ == "__main__":

    # For dev workers, deploy to Process-type dev work pool.
    # Using flow.deploy (per https://github.com/PrefectHQ/prefect/pull/13981), despite
    # reccomendations that flow.serve is better for development, on the hope that
    # it will be easier to use the same logic for both dev and prod. However I'm probably
    # wrong about that and just don't know it yet.
    etl.from_source(
        source="/opt/prefect/flows/",
        entrypoint="etl_flow.py:etl",
    ).deploy(
        name="etl",
        work_pool_name="lf-dev",
    )
