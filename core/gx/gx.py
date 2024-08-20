import great_expectations as gx
from core.gx.suites import EXPECTATION_SUITES

GX_DATASOURCE_NAME = 'pandas_datasource'

# Maintain a singleton GX EphemeralDataContext
gx_ctx = None


# TODO: Switch to FileDataContext instead of ephermeral?
def get_gx_context():
    global gx_ctx
    if gx_ctx is None:
        gx_ctx = initialize_gx()
    return gx_ctx


def initialize_gx():
    """Set up the great expectations context and expectation suites"""
    gx_ctx = gx.get_context()
    gx_ctx.sources.add_pandas(name=GX_DATASOURCE_NAME)

    for s in EXPECTATION_SUITES:
        suite = gx_ctx.add_expectation_suite(expectation_suite_name=s['name'])
        for exp_cfg in s['exp_cfgs']:
            suite.add_expectation(expectation_configuration=exp_cfg)
        gx_ctx.save_expectation_suite(expectation_suite=suite)

    return gx_ctx


def run_gx_checkpoint(suite_name, df):
    gx_ctx = get_gx_context()
    datasource = gx_ctx.get_datasource(GX_DATASOURCE_NAME)
    data_asset = datasource.add_dataframe_asset(name=f'{suite_name}-df')
    batch_request = data_asset.build_batch_request(dataframe=df)
    checkpoint_name = f'{suite_name}-checkpoint'
    checkpoint = gx_ctx.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                'batch_request': batch_request,
                'expectation_suite_name': suite_name,
            },
        ],
    )
    result = checkpoint.run(result_format='BASIC')
    if not result['success']:
        # TODO: Something more robust. Log event to datadog for monitoring?
        # Generate data doc artifact?
        print(f'GX Checkpoint failure: checkpoint_name:{checkpoint_name}')
        print(result.list_validation_results())
    else:
        print(f'GX Checkpoint success: checkpoint_name:{checkpoint_name}')
