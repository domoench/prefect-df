from flows.etl_flow import etl
from flows.train_model_flow import train_model
from flows.etl_and_train_flow import etl_and_train
from flows.compare_models_flow import compare_models
from flows.test_flow import test_flow


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

    # TODO remove
    test_flow.from_source(
        source="/opt/prefect/flows/",
        entrypoint="test_flow.py:test_flow",
    ).deploy(
        name="test_flow",
        work_pool_name="lf-dev",
    )

    print('Flows deployed to pool lf-dev.')
