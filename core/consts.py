EIA_EARLIEST_HOUR_UTC = '2015-07-01 05:00:00+00:00'

# Number of hours to give EIA time to collect data from balancing authorities
EIA_BUFFER_HOURS = 2 * 24

# Ensure the number of hours available to the evaluation set (to be excluded
# from training.
EIA_TEST_SET_HOURS = 2 * 7 * 24
