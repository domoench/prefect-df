from core.types import validate_call
import datetime

# The following dict describes all national US holidays from 2018 through 2039.
# Cached from the Calendarific API
holidays_dict = {
    datetime.date(2018, 1, 1): "New Year's Day",
    datetime.date(2018, 1, 15): 'Martin Luther King Jr. Day',
    datetime.date(2018, 2, 19): "Presidents' Day",
    datetime.date(2018, 5, 28): 'Memorial Day',
    datetime.date(2018, 7, 4): 'Independence Day',
    datetime.date(2018, 9, 3): 'Labor Day',
    datetime.date(2018, 10, 8): 'Columbus Day',
    datetime.date(2018, 11, 11): 'Veterans Day',
    datetime.date(2018, 11, 12): 'Veterans Day (substitute)',
    datetime.date(2018, 11, 22): 'Thanksgiving Day',
    datetime.date(2018, 12, 24): 'Christmas Eve',
    datetime.date(2018, 12, 25): 'Christmas Day',
    datetime.date(2019, 1, 1): "New Year's Day",
    datetime.date(2019, 1, 21): 'Martin Luther King Jr. Day',
    datetime.date(2019, 2, 18): "Presidents' Day",
    datetime.date(2019, 5, 27): 'Memorial Day',
    datetime.date(2019, 7, 4): 'Independence Day',
    datetime.date(2019, 9, 2): 'Labor Day',
    datetime.date(2019, 10, 14): 'Columbus Day',
    datetime.date(2019, 11, 11): 'Veterans Day',
    datetime.date(2019, 11, 28): 'Thanksgiving Day',
    datetime.date(2019, 12, 24): 'Christmas Eve',
    datetime.date(2019, 12, 25): 'Christmas Day',
    datetime.date(2020, 1, 1): "New Year's Day",
    datetime.date(2020, 1, 20): 'Martin Luther King Jr. Day',
    datetime.date(2020, 2, 17): "Presidents' Day",
    datetime.date(2020, 5, 25): 'Memorial Day',
    datetime.date(2020, 7, 3): 'Independence Day (substitute)',
    datetime.date(2020, 7, 4): 'Independence Day',
    datetime.date(2020, 9, 7): 'Labor Day',
    datetime.date(2020, 10, 12): 'Columbus Day',
    datetime.date(2020, 11, 11): 'Veterans Day',
    datetime.date(2020, 11, 26): 'Thanksgiving Day',
    datetime.date(2020, 12, 24): 'Christmas Eve',
    datetime.date(2020, 12, 25): 'Christmas Day',
    datetime.date(2021, 1, 1): "New Year's Day",
    datetime.date(2021, 1, 18): 'Martin Luther King Jr. Day',
    datetime.date(2021, 1, 20): 'Inauguration Day',
    datetime.date(2021, 2, 15): "Presidents' Day",
    datetime.date(2021, 5, 31): 'Memorial Day',
    datetime.date(2021, 6, 18): 'Juneteenth (substitute)',
    datetime.date(2021, 6, 19): 'Juneteenth',
    datetime.date(2021, 7, 4): 'Independence Day',
    datetime.date(2021, 7, 5): 'Independence Day (substitute)',
    datetime.date(2021, 9, 6): 'Labor Day',
    datetime.date(2021, 10, 11): 'Columbus Day',
    datetime.date(2021, 11, 11): 'Veterans Day',
    datetime.date(2021, 11, 25): 'Thanksgiving Day',
    datetime.date(2021, 12, 24): 'Christmas Day (substitute)',
    datetime.date(2021, 12, 25): 'Christmas Day',
    datetime.date(2021, 12, 31): "New Year's Day (substitute)",
    datetime.date(2022, 1, 1): "New Year's Day",
    datetime.date(2022, 1, 17): 'Martin Luther King Jr. Day',
    datetime.date(2022, 2, 21): "Presidents' Day",
    datetime.date(2022, 5, 30): 'Memorial Day',
    datetime.date(2022, 6, 19): 'Juneteenth',
    datetime.date(2022, 6, 20): 'Juneteenth (substitute)',
    datetime.date(2022, 7, 4): 'Independence Day',
    datetime.date(2022, 9, 5): 'Labor Day',
    datetime.date(2022, 10, 10): 'Columbus Day',
    datetime.date(2022, 11, 11): 'Veterans Day',
    datetime.date(2022, 11, 24): 'Thanksgiving Day',
    datetime.date(2022, 12, 25): 'Christmas Day',
    datetime.date(2022, 12, 26): 'Christmas Day (substitute)',
    datetime.date(2023, 1, 1): "New Year's Day",
    datetime.date(2023, 1, 2): "New Year's Day (substitute)",
    datetime.date(2023, 1, 16): 'Martin Luther King Jr. Day',
    datetime.date(2023, 2, 20): "Presidents' Day",
    datetime.date(2023, 5, 29): 'Memorial Day',
    datetime.date(2023, 6, 19): 'Juneteenth',
    datetime.date(2023, 7, 4): 'Independence Day',
    datetime.date(2023, 9, 4): 'Labor Day',
    datetime.date(2023, 10, 9): 'Columbus Day',
    datetime.date(2023, 11, 10): 'Veterans Day (substitute)',
    datetime.date(2023, 11, 11): 'Veterans Day',
    datetime.date(2023, 11, 23): 'Thanksgiving Day',
    datetime.date(2023, 12, 25): 'Christmas Day',
    datetime.date(2024, 1, 1): "New Year's Day",
    datetime.date(2024, 1, 15): 'Martin Luther King Jr. Day',
    datetime.date(2024, 2, 19): "Presidents' Day",
    datetime.date(2024, 5, 27): 'Memorial Day',
    datetime.date(2024, 6, 19): 'Juneteenth',
    datetime.date(2024, 7, 4): 'Independence Day',
    datetime.date(2024, 9, 2): 'Labor Day',
    datetime.date(2024, 10, 14): 'Columbus Day',
    datetime.date(2024, 11, 11): 'Veterans Day',
    datetime.date(2024, 11, 28): 'Thanksgiving Day',
    datetime.date(2024, 12, 25): 'Christmas Day',
    datetime.date(2025, 1, 1): "New Year's Day",
    datetime.date(2025, 1, 20): 'Inauguration Day',
    datetime.date(2025, 2, 17): "Presidents' Day",
    datetime.date(2025, 5, 26): 'Memorial Day',
    datetime.date(2025, 6, 19): 'Juneteenth',
    datetime.date(2025, 7, 4): 'Independence Day',
    datetime.date(2025, 9, 1): 'Labor Day',
    datetime.date(2025, 10, 13): 'Columbus Day',
    datetime.date(2025, 11, 11): 'Veterans Day',
    datetime.date(2025, 11, 27): 'Thanksgiving Day',
    datetime.date(2025, 12, 25): 'Christmas Day',
    datetime.date(2026, 1, 1): "New Year's Day",
    datetime.date(2026, 1, 19): 'Martin Luther King Jr. Day',
    datetime.date(2026, 2, 16): "Presidents' Day",
    datetime.date(2026, 5, 25): 'Memorial Day',
    datetime.date(2026, 6, 19): 'Juneteenth',
    datetime.date(2026, 7, 3): 'Independence Day (substitute)',
    datetime.date(2026, 7, 4): 'Independence Day',
    datetime.date(2026, 9, 7): 'Labor Day',
    datetime.date(2026, 10, 12): 'Columbus Day',
    datetime.date(2026, 11, 11): 'Veterans Day',
    datetime.date(2026, 11, 26): 'Thanksgiving Day',
    datetime.date(2026, 12, 25): 'Christmas Day',
    datetime.date(2027, 1, 1): "New Year's Day",
    datetime.date(2027, 1, 18): 'Martin Luther King Jr. Day',
    datetime.date(2027, 2, 15): "Presidents' Day",
    datetime.date(2027, 5, 31): 'Memorial Day',
    datetime.date(2027, 6, 18): 'Juneteenth (substitute)',
    datetime.date(2027, 6, 19): 'Juneteenth',
    datetime.date(2027, 7, 4): 'Independence Day',
    datetime.date(2027, 7, 5): 'Independence Day (substitute)',
    datetime.date(2027, 9, 6): 'Labor Day',
    datetime.date(2027, 10, 11): 'Columbus Day',
    datetime.date(2027, 11, 11): 'Veterans Day',
    datetime.date(2027, 11, 25): 'Thanksgiving Day',
    datetime.date(2027, 12, 24): 'Christmas Day (substitute)',
    datetime.date(2027, 12, 25): 'Christmas Day',
    datetime.date(2027, 12, 31): "New Year's Day (substitute)",
    datetime.date(2028, 1, 1): "New Year's Day",
    datetime.date(2028, 1, 17): 'Martin Luther King Jr. Day',
    datetime.date(2028, 2, 21): "Presidents' Day",
    datetime.date(2028, 5, 29): 'Memorial Day',
    datetime.date(2028, 6, 19): 'Juneteenth',
    datetime.date(2028, 7, 4): 'Independence Day',
    datetime.date(2028, 9, 4): 'Labor Day',
    datetime.date(2028, 10, 9): 'Columbus Day',
    datetime.date(2028, 11, 10): 'Veterans Day (substitute)',
    datetime.date(2028, 11, 11): 'Veterans Day',
    datetime.date(2028, 11, 23): 'Thanksgiving Day',
    datetime.date(2028, 12, 25): 'Christmas Day',
    datetime.date(2029, 1, 1): "New Year's Day",
    datetime.date(2029, 1, 15): 'Martin Luther King Jr. Day',
    datetime.date(2029, 1, 20): 'Inauguration Day',
    datetime.date(2029, 2, 19): "Presidents' Day",
    datetime.date(2029, 5, 28): 'Memorial Day',
    datetime.date(2029, 6, 19): 'Juneteenth',
    datetime.date(2029, 7, 4): 'Independence Day',
    datetime.date(2029, 9, 3): 'Labor Day',
    datetime.date(2029, 10, 8): 'Columbus Day',
    datetime.date(2029, 11, 11): 'Veterans Day',
    datetime.date(2029, 11, 12): 'Veterans Day (substitute)',
    datetime.date(2029, 11, 22): 'Thanksgiving Day',
    datetime.date(2029, 12, 25): 'Christmas Day',
    datetime.date(2030, 1, 1): "New Year's Day",
    datetime.date(2030, 1, 21): 'Martin Luther King Jr. Day',
    datetime.date(2030, 2, 18): "Presidents' Day",
    datetime.date(2030, 5, 27): 'Memorial Day',
    datetime.date(2030, 6, 19): 'Juneteenth',
    datetime.date(2030, 7, 4): 'Independence Day',
    datetime.date(2030, 9, 2): 'Labor Day',
    datetime.date(2030, 10, 14): 'Columbus Day',
    datetime.date(2030, 11, 11): 'Veterans Day',
    datetime.date(2030, 11, 28): 'Thanksgiving Day',
    datetime.date(2030, 12, 25): 'Christmas Day',
    datetime.date(2031, 1, 1): "New Year's Day",
    datetime.date(2031, 1, 20): 'Martin Luther King Jr. Day',
    datetime.date(2031, 2, 17): "Presidents' Day",
    datetime.date(2031, 5, 26): 'Memorial Day',
    datetime.date(2031, 6, 19): 'Juneteenth',
    datetime.date(2031, 7, 4): 'Independence Day',
    datetime.date(2031, 9, 1): 'Labor Day',
    datetime.date(2031, 10, 13): 'Columbus Day',
    datetime.date(2031, 11, 11): 'Veterans Day',
    datetime.date(2031, 11, 27): 'Thanksgiving Day',
    datetime.date(2031, 12, 25): 'Christmas Day',
    datetime.date(2032, 1, 1): "New Year's Day",
    datetime.date(2032, 1, 19): 'Martin Luther King Jr. Day',
    datetime.date(2032, 2, 16): "Presidents' Day",
    datetime.date(2032, 5, 31): 'Memorial Day',
    datetime.date(2032, 6, 18): 'Juneteenth (substitute)',
    datetime.date(2032, 6, 19): 'Juneteenth',
    datetime.date(2032, 7, 4): 'Independence Day',
    datetime.date(2032, 7, 5): 'Independence Day (substitute)',
    datetime.date(2032, 9, 6): 'Labor Day',
    datetime.date(2032, 10, 11): 'Columbus Day',
    datetime.date(2032, 11, 11): 'Veterans Day',
    datetime.date(2032, 11, 25): 'Thanksgiving Day',
    datetime.date(2032, 12, 24): 'Christmas Day (substitute)',
    datetime.date(2032, 12, 25): 'Christmas Day',
    datetime.date(2032, 12, 31): "New Year's Day (substitute)",
    datetime.date(2033, 1, 1): "New Year's Day",
    datetime.date(2033, 1, 17): 'Martin Luther King Jr. Day',
    datetime.date(2033, 1, 20): 'Inauguration Day',
    datetime.date(2033, 2, 21): "Presidents' Day",
    datetime.date(2033, 5, 30): 'Memorial Day',
    datetime.date(2033, 6, 19): 'Juneteenth',
    datetime.date(2033, 6, 20): 'Juneteenth (substitute)',
    datetime.date(2033, 7, 4): 'Independence Day',
    datetime.date(2033, 9, 5): 'Labor Day',
    datetime.date(2033, 10, 10): 'Columbus Day',
    datetime.date(2033, 11, 11): 'Veterans Day',
    datetime.date(2033, 11, 24): 'Thanksgiving Day',
    datetime.date(2033, 12, 25): 'Christmas Day',
    datetime.date(2033, 12, 26): 'Christmas Day (substitute)',
    datetime.date(2034, 1, 1): "New Year's Day",
    datetime.date(2034, 1, 2): "New Year's Day (substitute)",
    datetime.date(2034, 1, 16): 'Martin Luther King Jr. Day',
    datetime.date(2034, 2, 20): "Presidents' Day",
    datetime.date(2034, 5, 29): 'Memorial Day',
    datetime.date(2034, 6, 19): 'Juneteenth',
    datetime.date(2034, 7, 4): 'Independence Day',
    datetime.date(2034, 9, 4): 'Labor Day',
    datetime.date(2034, 10, 9): 'Columbus Day',
    datetime.date(2034, 11, 10): 'Veterans Day (substitute)',
    datetime.date(2034, 11, 11): 'Veterans Day',
    datetime.date(2034, 11, 23): 'Thanksgiving Day',
    datetime.date(2034, 12, 25): 'Christmas Day',
    datetime.date(2035, 1, 1): "New Year's Day",
    datetime.date(2035, 1, 15): 'Martin Luther King Jr. Day',
    datetime.date(2035, 2, 19): "Presidents' Day",
    datetime.date(2035, 5, 28): 'Memorial Day',
    datetime.date(2035, 6, 19): 'Juneteenth',
    datetime.date(2035, 7, 4): 'Independence Day',
    datetime.date(2035, 9, 3): 'Labor Day',
    datetime.date(2035, 10, 8): 'Columbus Day',
    datetime.date(2035, 11, 11): 'Veterans Day',
    datetime.date(2035, 11, 12): 'Veterans Day (substitute)',
    datetime.date(2035, 11, 22): 'Thanksgiving Day',
    datetime.date(2035, 12, 25): 'Christmas Day',
    datetime.date(2036, 1, 1): "New Year's Day",
    datetime.date(2036, 1, 21): 'Martin Luther King Jr. Day',
    datetime.date(2036, 2, 18): "Presidents' Day",
    datetime.date(2036, 5, 26): 'Memorial Day',
    datetime.date(2036, 6, 19): 'Juneteenth',
    datetime.date(2036, 7, 4): 'Independence Day',
    datetime.date(2036, 9, 1): 'Labor Day',
    datetime.date(2036, 10, 13): 'Columbus Day',
    datetime.date(2036, 11, 11): 'Veterans Day',
    datetime.date(2036, 11, 27): 'Thanksgiving Day',
    datetime.date(2036, 12, 25): 'Christmas Day',
    datetime.date(2037, 1, 1): "New Year's Day",
    datetime.date(2037, 1, 19): 'Martin Luther King Jr. Day',
    datetime.date(2037, 1, 20): 'Inauguration Day',
    datetime.date(2037, 2, 16): "Presidents' Day",
    datetime.date(2037, 5, 25): 'Memorial Day',
    datetime.date(2037, 6, 19): 'Juneteenth',
    datetime.date(2037, 7, 3): 'Independence Day (substitute)',
    datetime.date(2037, 7, 4): 'Independence Day',
    datetime.date(2037, 9, 7): 'Labor Day',
    datetime.date(2037, 10, 12): 'Columbus Day',
    datetime.date(2037, 11, 11): 'Veterans Day',
    datetime.date(2037, 11, 26): 'Thanksgiving Day',
    datetime.date(2037, 12, 25): 'Christmas Day',
    datetime.date(2038, 1, 1): "New Year's Day",
    datetime.date(2038, 1, 18): 'Martin Luther King Jr. Day',
    datetime.date(2038, 2, 15): "Presidents' Day",
    datetime.date(2038, 5, 31): 'Memorial Day',
    datetime.date(2038, 6, 18): 'Juneteenth (substitute)',
    datetime.date(2038, 6, 19): 'Juneteenth',
    datetime.date(2038, 7, 4): 'Independence Day',
    datetime.date(2038, 7, 5): 'Independence Day (substitute)',
    datetime.date(2038, 9, 6): 'Labor Day',
    datetime.date(2038, 10, 11): 'Columbus Day',
    datetime.date(2038, 11, 11): 'Veterans Day',
    datetime.date(2038, 11, 25): 'Thanksgiving Day',
    datetime.date(2038, 12, 24): 'Christmas Day (substitute)',
    datetime.date(2038, 12, 25): 'Christmas Day',
    datetime.date(2038, 12, 31): "New Year's Day (substitute)",
    datetime.date(2039, 1, 1): "New Year's Day",
    datetime.date(2039, 1, 17): 'Martin Luther King Jr. Day',
    datetime.date(2039, 2, 21): "Presidents' Day",
    datetime.date(2039, 5, 30): 'Memorial Day',
    datetime.date(2039, 6, 19): 'Juneteenth',
    datetime.date(2039, 6, 20): 'Juneteenth (substitute)',
    datetime.date(2039, 7, 4): 'Independence Day',
    datetime.date(2039, 9, 5): 'Labor Day',
    datetime.date(2039, 10, 10): 'Columbus Day',
    datetime.date(2039, 11, 11): 'Veterans Day',
    datetime.date(2039, 11, 24): 'Thanksgiving Day',
    datetime.date(2039, 12, 25): 'Christmas Day',
    datetime.date(2039, 12, 26): 'Christmas Day (substitute)'
}

holidays_set = set(holidays_dict.keys())


@validate_call
def is_holiday(d: datetime.date):
    return d in holidays_set