"""
This module provides functions to interact with AWS S3, read Parquet files, and perform unit tests
for a web service. It includes functions to download Parquet files from S3, concatenate them,
and generate file paths based on a date pattern. Additionally, it contains unit tests for
the service's endpoints.
"""

import os
import unittest

import numpy as np
import pandas as pd
import requests

FLASK_API_URL = os.getenv("FLASK_API_URL")

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

        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
        df["pickup_datetime"] = df["pickup_datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )  # pylint: disable=line-too-long
        df["dropoff_datetime"] = df["dropoff_datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )  # pylint: disable=line-too-long

        json_df = df.to_dict(orient="records")
        json_data = {"data": json_df}
        req = requests.post(url=f"{FLASK_API_URL}/predict", json=json_data, timeout=30)
        if req.json()["result"] != "success":
            raise ValueError("Invalid response")


if __name__ == '__main__':
    unittest.main()
