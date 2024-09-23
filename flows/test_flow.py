from prefect import flow
from prefect.logging import get_run_logger
import os


@flow(log_prints=True)
def test_flow():
    logger = get_run_logger()
    node_name = os.uname()[1]
    logger.info(f'Node: {node_name}')


if __name__ == '__main__':
    test_flow()
