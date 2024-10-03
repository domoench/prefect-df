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
    gx_ctx.data_sources.add_pandas(name=GX_DATASOURCE_NAME)

    for s in EXPECTATION_SUITES:
        # suite = gx_ctx.suites.add(name=s['name'])
        suite = gx.ExpectationSuite(name=s['name'])
        gx_ctx.suites.add(suite)
        for expectation in s['expectations']:
            suite.add_expectation(expectation)

    return gx_ctx


# TODO rename to remove the word checkpoint
def run_gx_checkpoint(suite_name, df):
    gx_ctx = get_gx_context()

    # Connect Data
    data_source = gx_ctx.get_datasource(name=GX_DATASOURCE_NAME)
    df_asset_name = f'{suite_name}-df'
    data_asset = data_source.add_dataframe_asset(name=df_asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(f'{df_asset_name}-batch_def')

    # Create Validation Definition
    suite = gx_ctx.suites.get(name=suite_name)
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=f'{suite_name}-val'
    )
    results = validation_definition.run(batch_parameters={'dataframe': df})
    if not results['success']:
        # TODO: Something more robust. Log event to datadog for monitoring?
        # Generate data doc artifact?
        print(f'GX Validation failure. suite:{suite_name}')
        print(results.describe_dict())
    else:
        print(f'GX Validation success: suite:{suite_name}')
