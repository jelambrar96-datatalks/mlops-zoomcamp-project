import os


from io import BytesIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from typing import List


import requests

import boto3
import s3fs
import pyarrow.parquet as pq

import time
import random
import logging 
import uuid
import pytz
import numpy as np
import pandas as pd
import psycopg2

from prefect import task, flow
from prefect.client.schemas.schedules import IntervalSchedule
from prefect.deployments import Deployment

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
# SEND_TIMEOUT = 10



# Get environment variables
# START_TIME = os.getenv("START_TIME", "2023-01-01")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# SEND_TIMEOUT = int(os.getenv("SEND_TIMEOUT", "10"))
# WHILE_TRUE = os.getenv("WHILE_TRUE", "false").lower() == "true"

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")


print(f"POSTGRES_HOST: {POSTGRES_HOST}")
print(f"POSTGRES_PORT: {POSTGRES_PORT}")
print(f"POSTGRES_USERNAME: {POSTGRES_USERNAME}")
print(f"POSTGRES_PASSWORD: {POSTGRES_PASSWORD}")
print(f"POSTGRES_DATABASE: {POSTGRES_DATABASE}")



CREATE_TABLE_STATEMENT = """
create table if not exists dummy_metrics(
	timestamp timestamp,
	value1 integer,
	value2 varchar,
	value3 float
)
"""

FLASK_ENDPOINT_URL = os.getenv("FLASK_ENDPOINT_URL", "http://flask-app:8000")
S3_ENDPOINT_URL =    os.getenv("S3_ENDPOINT_URL", "http://localstack:4566")



report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])


numerical_cols = [
    "pickup_minutes",
    "trip_distance"
]

categorical_cols = [
    "PULocationID",
    "DOLocationID",
    "pickup_weekday",
]
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=numerical_cols,
    categorical_features=categorical_cols,
    target=None
)



# Create a session to interact with LocalStack
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Configure LocalStack endpoint
s3_client = session.client(
    service_name='s3',
    endpoint_url=S3_ENDPOINT_URL,
)

@task
def prep_db():
    """
    create a sql table
    """
    # Construct the PostgreSQL connection string using the provided global variables.
    str_connection = "host={host} port={port} dbname={dbname} user={user} password={password}"
    str_connection = str_connection.format(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DATABASE,
        user=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD
    )
    print(str_connection)
    # Establish the PostgreSQL connection with autocommit enabled.
    with psycopg2.connect(str_connection) as conn:
        conn.autocommit = True
        with conn.cursor() as curr:
            curr.execute(CREATE_TABLE_STATEMENT)


def read_parquet_from_s3(s3, bucket, key):
    """
    Downloads a Parquet file from S3 and returns it as a pandas DataFrame.

    Args:
        s3: The S3 client object.
        bucket (str): The name of the S3 bucket.
        key (str): The key of the Parquet file in S3.

    Returns:
        pd.DataFrame: The Parquet file content as a pandas DataFrame.
    """
    response = s3.get_object(Bucket=bucket, Key=key)
    file_stream = BytesIO(response['Body'].read())
    table = pq.read_table(file_stream)
    df = table.to_pandas()
    return df


@task
def get_parquet_files(
        type_tripdata,
        start_datetime,
        end_datetime,
        bucket_name,
        s3_object,
        sample=None
        ) -> pd.DataFrame:
    """
    Concatenates all Parquet files from S3 for the given time range and trip type.

    Args:
        type_tripdata (str): The type of trip data (e.g., 'yellow').
        start_datetime (datetime): The start date and time.
        end_datetime (datetime): The end date and time.
        bucket_name (str): The S3 bucket name.
        s3_object: The S3 client object.
        sample (int, optional): Number of rows to sample from each DataFrame.

    Returns:
        pd.DataFrame: The concatenated DataFrame of all Parquet files.
    """
    output_df = None
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        current_year = current_datetime.year
        current_month = current_datetime.month
        delta = relativedelta(year=current_year, month=current_month, months=1)
        current_datetime = current_datetime + delta
        key_file = (f"nyc-taxi-data/type={type_tripdata}/year={current_year:04d}/"
                    f"month={current_month:02d}/raw-data.parquet")
        try:
            current_df = read_parquet_from_s3(bucket=bucket_name, key=key_file, s3=s3_object)
        except Exception:  # pylint: disable=broad-except
            continue
        if sample is not None:
            if current_df.shape[0] > sample:
                current_df = current_df.sample(sample, random_state=1)
        if output_df is None:
            output_df = current_df.copy()
            continue
        output_df = pd.concat((output_df, current_df))
    if sample is not None and output_df is not None:
        if output_df.shape[0] > sample:
            output_df = output_df.sample(sample, random_state=1)
    return output_df


# -----------------------------------------------------------------------------

@task
def prepare_df(df):
    """
    Prepares a DataFrame by performing several transformations on its columns. 

    The function modifies the DataFrame in the following ways:
    1. Converts the 'pickup_datetime' column to a datetime object.
    2. Extracts the day of the week from 'pickup_datetime' and stores it as a string in a new
    column 'pickup_weekday'.
    3. Computes the pickup time in minutes since midnight and stores it in a new column
    'pickup_minutes'.
    4. Prefixes 'PULocationID' and 'DOLocationID' columns with 'PU_' and 'DO_', respectively.
    5. Filters the DataFrame to include only the columns specified in the lists `categorical_cols`
    and `numerical_cols`.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to be transformed. It must have the columns:
        - 'pickup_datetime': Datetime of the pickup.
        - 'PULocationID': ID of the pickup location.
        - 'DOLocationID': ID of the dropoff location.

    Returns:
    --------
    pd.DataFrame
        The transformed DataFrame with only the specified categorical and numerical columns.
    
    Notes:
    ------
    - The lists `categorical_cols` and `numerical_cols` should be defined elsewhere in your code.
      These lists determine the columns that will be retained in the final DataFrame.
    - Assumes that 'pickup_datetime' is in a format recognized by `pd.to_datetime`.
    - The column 'pickup_minutes' represents the time in minutes from midnight (0:00 AM) for
      the pickup time.
    
    Raises:
    -------
    KeyError:
        If the input DataFrame `df` does not contain the required columns.
    """
    # Convert 'pickup_datetime' column to datetime objects.
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    # Extract the day of the week as a string from 'pickup_datetime' and create a new column
    # 'pickup_weekday'.
    df["pickup_weekday"] = df["pickup_datetime"].apply(lambda x: str(x.weekday()))
    # Compute the pickup time in minutes from midnight and store in 'pickup_minutes'.
    df["pickup_minutes"] = df["pickup_datetime"].apply(lambda x: x.hour * 60 + x.minute)
    # Prefix 'PULocationID' and 'DOLocationID' with 'PU_' and 'DO_' respectively.
    df["PULocationID"] = "PU_" + df["PULocationID"]
    df["DOLocationID"] = "DO_" + df["DOLocationID"]
    # Return the transformed DataFrame.
    return df


@task
def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the DataFrame by cleaning and formatting its columns.

    The function performs the following operations:
    1. Replaces string 'nan' with actual NaN values.
    2. Replaces all NaN values with 0.
    3. Formats the 'pickup_datetime' and 'dropoff_datetime' columns to string format
       '%Y-%m-%d %H:%M:%S'.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to be cleaned and formatted. The DataFrame
        must include columns 'pickup_datetime' and 'dropoff_datetime'.

    Returns:
    --------
    pd.DataFrame
        The cleaned and formatted DataFrame.
    """
    df.replace('nan', np.nan, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    df["pickup_datetime"] = df["pickup_datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))  # pylint: disable=line-too-long
    df["dropoff_datetime"] = df["dropoff_datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))  # pylint: disable=line-too-long
    return df


@task
def send_to_api(df: pd.DataFrame) -> List:
    """
    Sends the DataFrame records to an API endpoint as a JSON payload.

    The function performs the following operations:
    1. Converts the DataFrame into a list of dictionaries.
    2. Wraps the list in a JSON object under the key "data".
    3. Sends a POST request to the predefined API endpoint.
    4. Returns the predictions from the API response.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to be converted and sent to the API.

    Returns:
    --------
    List
        A list of predictions returned by the API, or None if the request failed.
    """
    json_df = df.to_dict(orient="records")
    json_data = {
        "data": json_df
    }

    res = None
    try:
        res = requests.post(FLASK_ENDPOINT_URL, json=json_data, timeout=50)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)

    if res is None:
        return None

    res_json = res.json()
    if res_json["result"] == "success":
        predictions = res_json["predictions"]
        return predictions
    return None


@task
def send_data(prediction_drift: float, num_drifted_columns: int, missing_values: float) -> None:
    """
    Sends a report to the PostgreSQL database.

    This function connects to a PostgreSQL database and inserts a new record into the
    'dummy_metrics' table, which includes the timestamp, prediction drift, number of drifted
    columns, and the share of missing values.

    Parameters:
    -----------
    prediction_drift : float
        The prediction drift value to be reported.

    num_drifted_columns : int
        The number of columns that have drifted.

    missing_values : float
        The share of missing values in the data.

    Returns:
    --------
    None
    """
    # Construct the PostgreSQL connection string using the provided global variables.
    str_connection = "host={host} port={port} dbname={dbname} user={user} password={password}"
    str_connection = str_connection.format(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DATABASE,
        user=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD
    )
    # Establish the PostgreSQL connection with autocommit enabled.
    with psycopg2.connect(str_connection) as conn:
        conn.autocommit = True
        # Open a new cursor to execute the metric calculation.
        with conn.cursor() as curr:
            curr.execute(
                "insert into dummy_metrics(timestamp, prediction_drift, "
                "num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
                (
                    datetime.now(),
                    prediction_drift,
                    num_drifted_columns,
                    missing_values
                )
            )



@flow
def batch_monitoring_backfill():
    """
    Executes a batch monitoring backfill process by connecting to a PostgreSQL database
    and calculating metrics for a set of predefined ranges. 
    The function manages the timing of the operations to ensure that data is sent
    at regular intervals, respecting a timeout period.

    The process involves:
    - Connecting to the PostgreSQL database using the provided connection details.
    - Enforcing a minimum time between data send operations, utilizing a sleep to manage timing.
    - Logging each successful data send operation.

    Global Variables:
    - POSTGRES_HOST (str): The host address of the PostgreSQL server.
    - POSTGRES_PORT (str): The port number of the PostgreSQL server.
    - POSTGRES_DATABASE (str): The name of the PostgreSQL database.
    - POSTGRES_USERNAME (str): The username for the PostgreSQL connection.
    - POSTGRES_PASSWORD (str): The password for the PostgreSQL connection.
    - SEND_TIMEOUT (int): The minimum time interval between consecutive data send operations.

    Raises:
    - psycopg.Error: If there is an issue with the PostgreSQL connection or operations.
    - Exception: For any other general issues that might arise during the process.
    """

    prep_db()

    reference_data = get_parquet_files(
            type_tripdata="yellow",
            start_datetime=datetime.strptime("2024-01-01", "%Y-%m-%d"),
            end_datetime=datetime.now(),
            bucket_name=S3_BUCKET_NAME,
            s3_object=s3_client,
            sample=1e6
        )
    if reference_data is None:
        return
    reference_data = prepare_columns(reference_data)
    reference_data = prepare_df(reference_data)
    # reference_predictions = send_to_api(reference_data)


    df = get_parquet_files(
            type_tripdata="yellow",
            start_datetime=datetime.strptime("2024-01-01", "%Y-%m-%d"),
            end_datetime=datetime.now(),
            bucket_name=S3_BUCKET_NAME,
            s3_object=s3_client,
            sample=1
        )
    if df is None:
        return
    df = prepare_columns(df)
    df = prepare_df(df)
    predicts = send_to_api(df)
    df["prediction"] = pd.Series(predicts)

    report.run(
        reference_data=reference_data,
        current_data = df,
        column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    # TASK 2
    send_data(prediction_drift,
                    num_drifted_columns,
                    missing_values)


    # Log that data has been sent successfully.
    logging.info("data sent")


if __name__ == '__main__':

    # # Definir un horario de ejecución cada minuto
    # schedule = IntervalSchedule(interval=timedelta(minutes=1))
    # # build
    # deployment = Deployment.build_from_flow(
    #     flow=batch_monitoring_backfill,
    #     name="batch_monitoring_backfill_per_minute",
    #     schedule=schedule,
    #     work_queue_name="default"
    # )
    # # Guardar el deployment para luego ejecutarlo
    # deployment.apply()

    while True:
        batch_monitoring_backfill()
        time.sleep(30)
