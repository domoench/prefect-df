import io
from core.logging import get_logger

lg = get_logger()


# TODO Move this and other util functions to core package?
def print_df_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    lg.info(f'Dataframe info:\n{buffer.getvalue()}\n')
    lg.info(f'Dataframe dtypes:\n{df.dtypes}\n')
    lg.info(f'Dataframe head:\n{df.head()}\n')
