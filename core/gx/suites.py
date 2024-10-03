from core.consts import EIA_MAX_D_VAL, EIA_MIN_D_VAL, TIME_FEATURES, TARGET
import great_expectations as gx

# Import custom expectations to ensure they are registered
from core.gx.custom.expect_column_values_on_the_hour import ExpectColumnValuesOnTheHour

EXPECTATION_SUITES = [
    {
        # Expectations on the dataframe pulled from the ETL warehouse
        'name': 'etl',
        'expectations': [
            gx.expectations.ExpectColumnToExist(
                column='D',
            ),
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column='D',
                mostly=0.9,
            ),
            ExpectColumnValuesOnTheHour(
                column='utc_ts',
                mostly=1.0,
            ),
        ],
    },

    # Expectations on the pre-processed dataframe (that will serve as training
    # data to the xgboost model).
    {
        'name': 'train',
        'expectations': [
            gx.expectations.ExpectTableColumnsToMatchSet(
                column_set=['utc_ts'] + TIME_FEATURES + [TARGET],
                exact_match=False,
            ),
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column='D',
                mostly=1.0,
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column='D',
                min_value=EIA_MIN_D_VAL,
                max_value=EIA_MAX_D_VAL,
            ),
            gx.expectations.ExpectColumnValuesToBeUnique(
                column='utc_ts',
            ),
            ExpectColumnValuesOnTheHour(
                column='utc_ts',
                mostly=1.0,
            ),
            # TODO validate the number of entries based on the start+end timestamps
        ],
    }
]
