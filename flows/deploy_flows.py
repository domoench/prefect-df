from etl_flow import etl
from train_model_flow import train_model
from etl_and_train_flow import etl_and_train
from compare_models_flow import compare_models


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

    etl_and_train.from_source(
        source="/opt/prefect/flows/",
        entrypoint="etl_and_train_flow.py:etl_and_train",
    ).deploy(
        name="etl_and_train",
        work_pool_name="lf-dev",
    )

    compare_models.from_source(
        source="/opt/prefect/flows/",
        entrypoint="compare_models_flow.py:compare_models",
    ).deploy(
        name="compare_models",
        work_pool_name="lf-dev",
    )
