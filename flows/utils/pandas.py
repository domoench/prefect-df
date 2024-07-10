import io


def print_df_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    print('Result dataframe info:\n{buffer.getvalue()}\n')
    print(f'Result dataframe dtypes:\n{df.dtypes}\n')
    print(f'Result dataframe head:\n{df.head()}\n')
