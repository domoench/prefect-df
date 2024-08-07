from prefect import flow, task
from prefect.tasks import task_input_hash
from utils.storage import (
    obj_key_with_timestamps,
    ensure_empty_dir,
    get_dvc_remote_repo_url,
    get_dvc_data_url,
)
from utils.pandas import print_df_summary
from consts import EIA_EARLIEST_HOUR_UTC
import requests
import pandas as pd
import os
from datetime import datetime
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo

EIA_MAX_REQUEST_ROWS = 5000


def request_EIA_data(start_ts, end_ts, offset, length=EIA_MAX_REQUEST_ROWS):
    print(f'Fetching API page. offset:{offset}. length:{length}')
    url = ('https://api.eia.gov/v2/electricity/rto/region-data/data/?'
           'frequency=hourly&data[0]=value&facets[respondent][]=PJM&'
           'sort[0][column]=period&sort[0][direction]=asc')

    # Use list of tuples instead of dict to allow duplicate params
    params = [
      ('offset', offset),
      ('length', length),
      ('api_key', os.environ['EIA_API_KEY']),
      ('start', start_ts.strftime('%Y-%m-%dT%H')),
      ('end', end_ts.strftime('%Y-%m-%dT%H')),
      ('facets[type][]', 'D'),
      ('facets[type][]', 'DF'),
    ]

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r


@task
def get_eia_data_as_df(start_ts, end_ts, offset, length=EIA_MAX_REQUEST_ROWS):
    r = request_EIA_data(start_ts, end_ts, offset, length)
    df = pd.DataFrame(r.json()['response']['data'])
    return df


# TODO Locally, the cache value disappears while the key remains. Leading to
# the errors like:
# ValueError: Path /root/.prefect/storage/fde6265e3819476eb8fbcb4f234dc9fc
# does not exist.
@task(cache_key_fn=task_input_hash)
def concurrent_fetch_EIA_data(start_ts, end_ts):
    time_span = end_ts - start_ts
    hours = int(time_span.total_seconds() / 3600)

    # Query EIA to determine exactly how many records match our time range
    r = request_EIA_data(start_ts, end_ts, 0)
    num_total_records = int(r.json()['response']['total'])
    print(f'Total records to fetch: {num_total_records}')

    # Calculate how many paginated API requests will be required to fetch all
    # the timeseries data
    num_full_requests = num_total_records // EIA_MAX_REQUEST_ROWS
    final_request_length = num_total_records % EIA_MAX_REQUEST_ROWS
    print((f'Fetching {hours} hours of data: {num_total_records} records.\n',
          f'Start: {start_ts}. End: {end_ts}'))
    print((f'Will make {num_full_requests} {EIA_MAX_REQUEST_ROWS}-length requests '
           f'and one {final_request_length}-length request.'))

    # Make the requests concurrently
    result_df_futures = []

    # Initiate the full-length requests
    for i in range(num_full_requests):
        offset = i * EIA_MAX_REQUEST_ROWS
        future = get_eia_data_as_df.submit(start_ts, end_ts, offset)
        result_df_futures.append(future)

    # Initiate the final request for the remainder records
    offset = num_full_requests * EIA_MAX_REQUEST_ROWS
    future = get_eia_data_as_df.submit(start_ts, end_ts, offset, final_request_length)
    result_df_futures.append(future)

    result_dfs = [future.result() for future in result_df_futures]
    api_df = pd.concat(result_dfs)
    return api_df


@task
def extract(start_ts: datetime, end_ts: datetime):
    print("Fetching EIA electricty demand timeseries.")

    # Calculate the number of rows to fetch from the API between start and end
    eia_df = concurrent_fetch_EIA_data(start_ts, end_ts)

    print_df_summary(eia_df)

    return eia_df


@task
def transform(eia_df: pd.DataFrame):
    print('Transforming timeseries.')

    # Convert types
    eia_df['UTC period'] = pd.to_datetime(eia_df['period'], utc=True)
    eia_df['value'] = pd.to_numeric(eia_df['value'])

    # Careful, EIA results can have duplicates (at the boundaries of the pages)
    # And such behavior seems to be non-deterministic.
    # Remove duplicates
    eia_df = eia_df.drop_duplicates(subset=['UTC period', 'value', 'type'])

    # In the EIA API response, for any given hour, there are between 0 and 2 records:
    # 1 record for D value, and another for the DF value. Update dataframe to have 1 row
    # for each hour, with 2 columns: D and DF. Units are MWh.
    demand_df = eia_df[eia_df.type == 'D']
    d_forecast_df = eia_df[eia_df.type == 'DF']

    # Create base dataframe with a timestamp for every hour in the range
    start_ts = eia_df['UTC period'].min()
    end_ts = eia_df['UTC period'].max()
    dt_df = pd.DataFrame({'utc_ts': pd.date_range(start=start_ts, end=end_ts, freq='h')})

    # Merge in the demand timeseries
    merge_df = pd.merge(
        dt_df,
        demand_df[["UTC period", "respondent", "value"]].rename(columns={"value": "D"}),
        left_on="utc_ts",
        right_on="UTC period",
        how="left",
    )

    # Merge in the demand forecast timeseries
    merge_df = pd.merge(
        merge_df,
        d_forecast_df[['UTC period', 'value']].rename(columns={'value': 'DF'}),
        left_on='utc_ts', right_on='UTC period',
        how='left'
    )

    merge_df = merge_df.drop(columns=['UTC period_x', 'UTC period_y'])

    # Set timestamp as index
    merge_df = merge_df.set_index('utc_ts')

    # TODO: Imputation? Or save that for later down the pipe?

    print_df_summary(merge_df)
    return merge_df


@task
def load_to_dvc(df: pd.DataFrame):
    print('Loading timeseries into warehouse.')

    # Ensure clean local git/dvc repo directory
    local_dvc_repo = os.getenv('DVC_LOCAL_REPO_PATH')
    ensure_empty_dir(local_dvc_repo)

    # Clone the git repo
    git_repo_url = get_dvc_remote_repo_url()

    # Initialize git and dvc python client instances
    git_repo = GitRepo.clone_from(git_repo_url, local_dvc_repo)
    dvc_repo = DvcRepo(local_dvc_repo)

    # Write dataset file into local directory
    start_ts = df.index.min()
    end_ts = df.index.max()
    filename = obj_key_with_timestamps('eia_d_df', start_ts, end_ts)
    dvc_dataset_path = f'data/{filename}'
    local_dataset_path = f'{local_dvc_repo}/{dvc_dataset_path}'
    df.to_parquet(local_dataset_path)

    # Add dataset to dvc tracking
    dvc_repo.add(local_dataset_path)

    # Git Tracking
    git_repo.git.add(f'{local_dataset_path}.dvc')
    git_repo.git.add(f'{local_dvc_repo}/data/.gitignore')

    diffs = git_repo.index.diff('HEAD')
    diff_files = list(map(lambda d: d.a_path, diffs))
    git_commit_hash = None
    if len(diff_files) > 0:
        print(f'Staged files:\n{diff_files}')

        commit_msg = 'Add dataset.'
        commit = git_repo.index.commit(commit_msg)
        tag_str = f"s{start_ts.strftime('%Y-%m-%d_%H')}.e{end_ts.strftime('%Y-%m-%d_%H')}"
        git_repo.create_tag(ref=commit, path=tag_str)
        git_commit_hash = str(commit)
        print(f'Git commit hash: {git_commit_hash}')
        git_repo.remote(name='origin').push()
        # TODO git tags?

        print('Pushing dataset to DVC remote storage')
        # Note: Push requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        dvc_repo.push()
    else:
        git_commit_hash = str(git_repo.head.commit)
        print(f'No dataset changes. Using HEAD as commit hash: {git_commit_hash}')

    dvc_data_info = {
        'repo': git_repo_url,
        'path': dvc_dataset_path,
        'rev': git_commit_hash,
    }
    return dvc_data_info


@flow(log_prints=True)
def etl(start_ts: datetime = pd.to_datetime(EIA_EARLIEST_HOUR_UTC).to_pydatetime(),
        end_ts: datetime = (pd.Timestamp.utcnow().round('h') - pd.Timedelta(weeks=1)).to_pydatetime()):
    """Pulls all available hourly EIA demand data between the given start and end
    timestamps and persists it in the DVC data warehouse as a parquet file.

    Returns:
        dict: A dict containing the info relevant to retrieve a dvc-tracked dataset
    """
    df = extract(start_ts, end_ts)
    df = transform(df)
    dvc_data_info = load_to_dvc(df)

    return dvc_data_info
