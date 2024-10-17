EIA_EARLIEST_HOUR_UTC = '2015-07-01 05:00:00+00:00'

# Number of hours to give EIA time to collect data from balancing authorities
EIA_BUFFER_HOURS = 2 * 24

# Ensure the number of hours available to the evaluation set (to be excluded
# from training.
EIA_TEST_SET_HOURS = 4 * 7 * 24

# Maximum number of rows to request in one call to the EIA API
EIA_MAX_REQUEST_ROWS = 5000

# Min and Max demand value caps, as determined from PJM demand data between 2015 and 2024
EIA_MAX_D_VAL = 165_000
EIA_MIN_D_VAL = 60_000


TIME_FEATURES = ['hour', 'month', 'year', 'quarter', 'dayofweek', 'dayofmonth', 'dayofyear']
TARGET = 'D'
