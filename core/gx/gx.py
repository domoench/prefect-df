import great_expectations as gx
from core.gx.suites import add_expectation_suites
from datetime import datetime

GX_DATASOURCE_NAME = 'pandas_datasource'

# Maintain a singleton GX EphemeralDataContext
gx_ctx = None


def get_gx_context():
    global gx_ctx
    if gx_ctx is None:
        gx_ctx = initialize_gx()
    return gx_ctx


def initialize_gx():
    """Set up the great expectations context and expectation suites"""
    print('Initializing great expectation data context and suites.')
    gx_ctx = gx.get_context(mode='ephemeral')
    gx_ctx.data_sources.add_pandas(name=GX_DATASOURCE_NAME)
    add_expectation_suites(gx_ctx)

    return gx_ctx


# TODO rename to remove the word checkpoint
def run_gx_checkpoint(suite_name, df):
    gx_ctx = get_gx_context()

    # Connect Data
    data_source = gx_ctx.get_datasource(name=GX_DATASOURCE_NAME)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_asset_name = f'{suite_name}-df-{timestamp}'
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
