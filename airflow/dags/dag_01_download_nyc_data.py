"""
Airflow DAG that makes an HTTP request,
converts the content to a DataFrame,
and then saves it as a .parquet file
"""


import os

from io import BytesIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import boto3
import pandas as pd
import requests
import requests.exceptions as rexcep
import s3fs # pylint: disable=unused-import

from airflow import DAG
from airflow.operators.dummy import DummyOperator # pylint: disable=import-error,no-name-in-module
from airflow.operators.python import PythonOperator


# get env vars
AIRFLOW_START_TIME = os.getenv("AIRFLOW_START_TIME", "2023-01-01")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


URL_PATTERN = "https://d37ci6vzurychx.cloudfront.net/trip-data/{type_tripdata}_tripdata_{str_download_date}.parquet" # pylint: disable=line-too-long
TIMEOUT_SECONDS = 30
MONTH_RELATIVE_DELTA = 3


# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


dag_01_download_nyc_data = DAG(
    'dag_01_download_nyc_data',
    default_args=default_args,
    description='A simple DAG to fetch data and store as Parquet',
    schedule_interval='0 0 1 * *', # At 00:00 on day-of-month 1.
    start_date=datetime.strptime(AIRFLOW_START_TIME, "%Y-%m-%d"),
    catchup=False,
)


task_start = DummyOperator(
    task_id="task_start",
    dag=dag_01_download_nyc_data
)


TRIPDATA_DICT = {
    'yellow': {
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'tpep_pickup_datetime': 'pickup_datetime'
    },
    'green': {
        'lpep_dropoff_datetime': 'dropoff_datetime',
        'lpep_pickup_datetime': 'pickup_datetime'
    }
}

CATEGORICAL_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "VendorID",
    "RatecodeID",
    "payment_type"
]

NUMERICAL_COLUMNS = [
    'congestion_surcharge',
    'extra', 'fare_amount',
    'improvement_surcharge',
    'mta_tax',
    'passenger_count',
    'store_and_fwd_flag',
    'tip_amount',
    'tolls_amount',
    'total_amount',
    'dropoff_datetime',
    'pickup_datetime',
    'trip_distance'
]


S3_ENDPOINT_URL = "http://localstack:4566" # taken from docker-compose.yaml

STORAGE_OPTIONS = {
    'key': AWS_ACCESS_KEY_ID,
    'secret': AWS_SECRET_ACCESS_KEY,
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

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


def function_download_data(
        download_date: str,
        month_delta: int = 0, type_tripdata: str = "yellow") -> None:
    """
    Function to fetch data and convert to DataFrame
    """
    if type_tripdata not in ("yellow", "green"):
        raise ValueError('ERROR: invalid type_tripdata parametrer, '
                         'must be "yellow", "green"')

    download_date = datetime.strptime(download_date, "%Y-%m-%d")
    delta = relativedelta(year=download_date.year, month=download_date.month, months=month_delta)
    download_date = download_date - delta
    str_download_date = download_date.strftime("%Y-%m")

    # Making an HTTP request to the API
    url = URL_PATTERN.format(
        type_tripdata=type_tripdata,
        str_download_date=str_download_date
    )
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
    except (rexcep.ConnectionError, rexcep.HTTPError, rexcep.Timeout,
            rexcep.TooManyRedirects, rexcep.RequestException) as exc:
        print(exc)
        return
    if response.status_code != 200:
        raise rexcep.RequestException(response.text)

    # decode content
    df = pd.read_parquet(BytesIO(response.content))
    # add metadata
    df["execution_date"] = str_download_date
    df["type_tripdata"] = type_tripdata

    # rename columns
    df = df.rename(columns=TRIPDATA_DICT[type_tripdata])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # remove no-common columns yellow-green
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype(str)
    all_columns = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
    df = df[all_columns]

    # create target column
    df["duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).apply(lambda x: x.total_seconds() / 60) # pylint: disable=line-too-long

    # cleaning data
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    # save on s3
    path_to_save = "s3://{s3_bucket}/nyc-taxi-data/type={type_tripdata}/year={year:04d}/month={month:02d}/raw-data.parquet" # pylint: disable=line-too-long
    path_to_save = path_to_save.format(
        s3_bucket=S3_BUCKET_NAME,
        type_tripdata=type_tripdata,
        year=download_date.year,
        month=download_date.month
    )
    df.to_parquet(
        path_to_save,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=STORAGE_OPTIONS
    )


task_download_data = PythonOperator(
    task_id='task_download_data',
    python_callable=function_download_data,
    dag=dag_01_download_nyc_data,
    op_kwargs={
        "download_date": "{{ ds }}",
        "month_delta": MONTH_RELATIVE_DELTA,
        "type_tripdata": "yellow"

    }
)



task_start >> task_download_data # pylint: disable=pointless-statement
