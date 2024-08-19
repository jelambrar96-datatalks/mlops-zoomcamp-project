"""
This module provides functions to interact with AWS S3, read Parquet files, and perform unit tests
for a web service. It includes functions to download Parquet files from S3, concatenate them,
and generate file paths based on a date pattern. Additionally, it contains unit tests for 
the service's endpoints.
"""

from io import BytesIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import boto3
import unittest
import requests
import pyarrow.parquet as pq
import numpy as np
import pandas as pd


# Get environment variables
AIRFLOW_START_TIME = os.getenv("AIRFLOW_START_TIME", "2023-01-01")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# S3 endpoint URL (from docker-compose.yaml)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
FLASK_API_URL = os.getenv("FLASK_API_URL")

session = None # pylint: disable=invalid-name
s3_client = None # pylint: disable=invalid-name
try:
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
finally:
    pass


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
        if sample is not None and current_df.shape[0] > sample:
            current_df = current_df.sample(sample, random_state=1)
        if output_df is None:
            output_df = current_df.copy()
            continue
        output_df = pd.concat((output_df, current_df))
    if sample is not None and output_df.shape[0] > sample:
        output_df = output_df.sample(sample, random_state=1)
    return output_df


class RequestsTest(unittest.TestCase):
    """
    Unit tests for the web service endpoints.
    """

    def test_index(self):
        """
        Test the index endpoint.
        """
        req = requests.get(FLASK_API_URL, timeout=10)
        if req.status_code != 200:
            raise ValueError(f"Invalid status code {req.status_code}")

    def test_01_reload(self):
        """
        Test the reload endpoint.
        """
        req = requests.post(f"{FLASK_API_URL}/reload", timeout=30)
        if req.status_code != 200:
            raise ValueError(f"Invalid status code {req.status_code}")
        if req.json()["result"] != "success":
            raise ValueError("Invalid response")

    def test_02_predict(self):
        """
        Test the predict endpoint.
        """
        df = pd.read_csv("./df.csv")
        df.replace('nan', np.nan, inplace=True)
        df.replace(np.nan, 0, inplace=True)

        df["pickup_datetime"]  = pd.to_datetime(df["pickup_datetime"])
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
        df["pickup_datetime"] = df["pickup_datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")) # pylint: disable=line-too-long
        df["dropoff_datetime"] = df["dropoff_datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")) # pylint: disable=line-too-long

        json_df = df.to_dict(orient="records")
        json_data = {
            "data": json_df
        }
        req = requests.post(
            url=f"{FLASK_API_URL}/predict",
            json=json_data,
            timeout=30
        )
        if req.json()["result"] != "success":
            raise ValueError("Invalid response")

if __name__ == '__main__':
    unittest.main()
