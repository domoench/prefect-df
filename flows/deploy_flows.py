from etl_flow import etl
from train_model_flow import train_model


if __name__ == "__main__":

    # For dev workers, deploy to Process-type dev work pool.
    # Using flow.deploy (per https://github.com/PrefectHQ/prefect/pull/13982),
    # despite reccomendations that flow.serve is better for development, on
    # the hope that it will be easier to use the same logic for both dev and
    # prod. However I'm probably wrong about that and just don't know it yet.
    etl.from_source(
        source="/opt/prefect/flows/",
        entrypoint="etl_flow.py:etl",
    ).deploy(
        name="etl",
        work_pool_name="lf-dev",
    )

    train_model.from_source(
        source="/opt/prefect/flows/",
        entrypoint="train_model_flow.py:train_model",
    ).deploy(
        name="train_model",
        work_pool_name="lf-dev",
    )
