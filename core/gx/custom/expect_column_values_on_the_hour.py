from typing import Optional

from great_expectations.expectations.expectation_configuration import ExpectationConfiguration
from great_expectations.exceptions import InvalidExpectationConfigurationError
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import ColumnMapExpectation
from great_expectations.expectations.metrics import (
    ColumnMapMetricProvider,
    column_condition_partial,
)


# This class defines a Metric to support your Expectation.
class ColumnValuesMatchSomeCriteria(ColumnMapMetricProvider):
    condition_metric_name = 'column_values.on_the_hour'

    # This method implements the core logic for the PandasExecutionEngine
    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column, **kwargs):
        return column.apply(lambda x: x.minute == 0 and x.second == 0)


class ExpectColumnValuesOnTheHour(ColumnMapExpectation):
    """Expect column values to be timestamps on the hour every hour."""

    # Id string of the Metric used by this Expectation.
    map_metric = 'column_values.on_the_hour'

    # This is a list of parameter names that can affect whether the Expectation evaluates to True or False
    success_keys = ('mostly',)

    # This dictionary contains default values for any parameters that should have default values
    default_kwarg_values = {}

    examples = [
        {
            'data': {
                'all_on_the_hour': [
                    '2023-01-01T00:00:00',
                    '2023-01-01T01:00:00',
                    '2023-01-01T02:00:00',
                ],
                'not_on_the_hour': [
                    '2023-01-01T00:01:00',
                    '2023-01-01T01:30:00',
                    '2023-01-01T02:45:00',
                ],
            },
            'tests': [
                {
                    'title': 'positive_test_with_all_on_the_hour',
                    'exact_match_out': False,
                    'in': {'column': 'all_on_the_hour'},
                    'out': {'success': True},
                },
                {
                    'title': 'negative_test_with_some_not_on_the_hour',
                    'exact_match_out': False,
                    'in': {'column': 'not_on_the_hour'},
                    'out': {'success': False},
                },
            ],
        }
    ]

    def validate_configuration(
        self, configuration: Optional[ExpectationConfiguration] = None
    ) -> None:
        """
        Validates that a configuration has been set, and sets a configuration if it has yet to be set. Ensures that
        necessary configuration arguments have been provided for the validation of the expectation.

        Args:
            configuration (OPTIONAL[ExpectationConfiguration]): \
                An optional Expectation Configuration entry that will be used to configure the expectation
        Returns:
            None. Raises InvalidExpectationConfigurationError if the config is not validated successfully
        """

        super().validate_configuration(configuration)
        configuration = configuration or self.configuration

        # # Check other things in configuration.kwargs and raise Exceptions if needed
        # try:
        #     assert (
        #         ...
        #     ), "message"
        #     assert (
        #         ...
        #     ), "message"
        # except AssertionError as e:
        #     raise InvalidExpectationConfigurationError(str(e))


if __name__ == '__main__':
    ExpectColumnValuesOnTheHour().print_diagnostic_checklist()
