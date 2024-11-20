from core.consts import (
    EIA_MAX_D_VAL, EIA_MIN_D_VAL, TARGET, TIME_FEATURES, LAG_FEATURES,
    HOLIDAY_FEATURES, WEATHER_FEATURES
)
import great_expectations as gx

# Import custom expectations to ensure they are registered
from core.gx.custom.expect_column_values_on_the_hour import ExpectColumnValuesOnTheHour


def add_expectation_suites(gx_ctx):
    add_dvc_expectation_suite(gx_ctx)
    add_train_expectation_suite(gx_ctx)


def add_dvc_expectation_suite(gx_ctx):
    # Expectations on the dataframe pushed to or pulled from the ETL warehouse
    suite = gx.ExpectationSuite(name='dvc')
    gx_ctx.suites.add(suite)

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=['D', 'temperature_2m', 'cloud_cover'],
            exact_match=True
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
    # or eval data to the xgboost model).
    suite = gx.ExpectationSuite(name='xgb_input')
    gx_ctx.suites.add(suite)

    features = TIME_FEATURES.copy() + LAG_FEATURES + HOLIDAY_FEATURES + WEATHER_FEATURES
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
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='cloud_cover',
            min_value=0,
            max_value=100,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column='cloud_cover',
            mostly=1.0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column='temperature_2m',
            mostly=1.0,
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
