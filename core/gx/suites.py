from core.consts import EIA_MAX_D_VAL, EIA_MIN_D_VAL, TIME_FEATURES, TARGET
from great_expectations.core.expectation_configuration import (
    ExpectationConfiguration,
)

# Import custom expectations to ensure they are registered
from core.gx.custom.expect_column_values_on_the_hour import ExpectColumnValuesOnTheHour


EXPECTATION_SUITES = [
    # Expectations on the dataframe pulled from the ETL warehouse
    {
        'name': 'etl',
        'exp_cfgs': [
            ExpectationConfiguration(
                expectation_type='expect_column_to_exist',
                kwargs={'column': 'D'},
            ),
            ExpectationConfiguration(
                expectation_type='expect_column_values_to_not_be_null',
                kwargs={'column': 'D', 'mostly': 0.9},
            ),
            ExpectationConfiguration(
                expectation_type='expect_column_values_on_the_hour',
                kwargs={'column': 'utc_ts', 'mostly': 1.0},
            ),
        ]
    },

    # Expectations on the pre-processed dataframe (that will serve as training
    # data to the xgboost model).
    {
        'name': 'train',
        'exp_cfgs': [
            ExpectationConfiguration(
                expectation_type='expect_table_columns_to_match_set',
                kwargs={
                    'column_set': ['utc_ts'] + TIME_FEATURES + [TARGET],
                    'exact_match': False,
                }
            ),
            ExpectationConfiguration(
                expectation_type='expect_column_values_to_not_be_null',
                kwargs={'column': 'D', 'mostly': 1.0},
            ),
            ExpectationConfiguration(
                expectation_type='expect_column_values_to_be_between',
                kwargs={'column': 'D', 'min_value': EIA_MIN_D_VAL, 'max_value': EIA_MAX_D_VAL},
            ),
            ExpectationConfiguration(
                expectation_type='expect_column_values_to_be_unique',
                kwargs={'column': 'utc_ts'},
            ),
            ExpectationConfiguration(
                expectation_type='expect_column_values_on_the_hour',
                kwargs={'column': 'utc_ts', 'mostly': 1.0},
            ),
        ]
    },
]
