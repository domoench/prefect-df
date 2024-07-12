import io


def print_df_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(f'Dataframe info:\n{buffer.getvalue()}\n')
    print(f'Dataframe dtypes:\n{df.dtypes}\n')
    print(f'Dataframe head:\n{df.head()}\n')
