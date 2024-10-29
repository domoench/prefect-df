from core.consts import EIA_MAX_D_VAL, EIA_MIN_D_VAL, TIME_FEATURES, TARGET
from core.model import get_model_features
from core.types import ModelFeatureFlags
import great_expectations as gx

# Import custom expectations to ensure they are registered
from core.gx.custom.expect_column_values_on_the_hour import ExpectColumnValuesOnTheHour


def add_expectation_suites(gx_ctx):
    add_etl_expectation_suite(gx_ctx)
    add_train_expectation_suite(gx_ctx)


def add_etl_expectation_suite(gx_ctx):
    # Expectations on the dataframe pulled from the ETL warehouse

    suite = gx.ExpectationSuite(name='etl')
    gx_ctx.suites.add(suite)
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(
            column='D'
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column='D',
            mostly=0.9,
        )
    )

    suite.add_expectation(
        ExpectColumnValuesOnTheHour(
            column='utc_ts',
            mostly=1.0,
        )
    )


def add_train_expectation_suite(gx_ctx):
    # Expectations on the pre-processed dataframe (that will serve as training
    # data to the xgboost model).
    suite = gx.ExpectationSuite(name='train')
    gx_ctx.suites.add(suite)

    features = get_model_features(ModelFeatureFlags(lag=True, weather=False, holidays=True))
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=['utc_ts'] + features + [TARGET],
            exact_match=True
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column='D',
            mostly=1.0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='D',
            min_value=EIA_MIN_D_VAL,
            max_value=EIA_MAX_D_VAL,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column='utc_ts',
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='is_holiday',
            value_set=[0, 1],
        )
    )

    suite.add_expectation(
        ExpectColumnValuesOnTheHour(
            column='utc_ts',
            mostly=1.0,
        )
    )
    # TODO validate the number of entries based on the start+end timestamps
