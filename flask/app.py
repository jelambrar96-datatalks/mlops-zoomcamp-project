"""
flask app
"""

import json
import os
import pickle

from datetime import datetime

import boto3

from flask import Flask, request, jsonify


import numpy
import pandas as pd
import sklearn
import mlflow

from sklearn.base import clone


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

MODEL_S3_PATH = os.getenv("MODEL_S3_PATH")
METADATA_S3_PATH = os.getenv("METADATA_S3_PATH")
DV_S3_PATH = os.getenv("DV_S3_PATH")


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


TEMP_MODEL_DIR = "/tmp/model/skmodel"
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)


class ModelLoader:
    """
    class save model
    """

    def __init__(self, *args, **kwargs):
        self._model = None
        self._dict_vectorizer = None
        self._metadata_model = None

    def reload(self):
        """
        load model
        """
        op = False
        try:
            model_local_path = f"{TEMP_MODEL_DIR}/model.pkl"
            print(MODEL_S3_PATH)
            s3_client.download_file(S3_BUCKET_NAME, MODEL_S3_PATH, model_local_path)
            self._model = pickle.load(open(model_local_path, 'rb'))
            # self._model = mlflow.sklearn.load_model(model_local_path)
            # print(type(self._model))
            dv_local_path = f"{TEMP_MODEL_DIR}/dict_vectorizer.pkl"
            s3_client.download_file(S3_BUCKET_NAME, DV_S3_PATH, dv_local_path)
            self._dict_vectorizer = pickle.load(open(dv_local_path, 'rb'))

            op = True
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)

        return op

    def predict(self, df_values: pd.DataFrame) -> list:
        """
        predict outputs model
        """
        if self._dict_vectorizer is None:
            return None
        # expected_features = self._dict_vectorizer.get_feature_names_out()
        # input_features = df_values.columns.tolist()
        # print("Expected Features:", expected_features)
        # print("Input Features:", input_features)
        try:
            x_values = self._dict_vectorizer.transform(df_values.to_dict(orient='records'))
            x_values = pd.DataFrame.sparse.from_spmatrix(x_values).sparse.to_dense()
            columns = dict( (item, str(i)) for i, item in enumerate(x_values.columns))
            x_values = x_values.rename(columns=columns)
            print(type(x_values))
            print(x_values.shape)
        except Exception as  e:
            print(e)
        if self._model is None:
            return None
        try:
            model_out = self._model.predict(x_values)
        except Exception as e:
            print(e)
            return None
        return model_out.tolist()

    def get_metadata(self):
        """
        get metadata from model
        """
        return self._metadata_model




def prepare_features(ride):

    if isinstance(ride, dict):
        ride = [ ride ]
    df = pd.DataFrame.from_records(ride)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_weekday"] = df["pickup_datetime"].apply(lambda x: str(x.weekday()))
    df["pickup_minutes"] = df["pickup_datetime"].apply(lambda x: x.hour + 60 * x.minute)
    df["PULocationID"] = "PU_" + df["PULocationID"]
    df["DOLocationID"] = "DO_" + df["DOLocationID"]

    numerical_cols = [
        "pickup_minutes",
        "trip_distance"
    ]

    categorical_cols = [
        "PULocationID",
        "DOLocationID",
        "pickup_weekday",
    ]

    df = df[ categorical_cols + numerical_cols ]

    return df


model_loader = ModelLoader()
app = Flask(__name__)


@app.route('/reload', methods=['POST'])
def reaload_endpoint():
    """
    Send sucess message if model reload is success
    """
    flag = model_loader.reload()
    if flag:
        return jsonify({"result": "success", "reloaded": True})
    return jsonify({"result": "failed", "reloaded": False})



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()["data"]
    features = prepare_features(ride)
    pred = model_loader.predict(features)
    result = "failed"
    if result is not None:
        result = "success"
    return jsonify({"result": result, "predictions": pred})


@app.route('/', methods=['GET'])
def index():
    """
    hello world app
    """
    return "Zoomcamp application"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
