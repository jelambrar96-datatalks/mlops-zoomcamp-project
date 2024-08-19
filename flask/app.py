"""
Flask application for serving machine learning predictions.

This application loads a machine learning model and serves predictions via a REST API.
It interacts with S3 for model storage and uses LocalStack for local development.
The model predicts outcomes based on ride data, such as pickup and dropoff locations.

Modules:
    json
    os
    pickle
    datetime
    boto3
    flask
    numpy
    pandas
    sklearn
    mlflow

Classes:
    ModelLoader

Functions:
    prepare_features(ride)
    reload_endpoint()
    predict_endpoint()
    index()

"""

import os
import pickle

import boto3

import numpy
import pandas as pd
import sklearn
import mlflow

from flask import Flask, request, jsonify


# Load AWS credentials and S3 bucket information from environment variables
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

# Create a temporary directory for storing the model and vectorizer locally
TEMP_MODEL_DIR = "/tmp/model/skmodel"
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

class ModelLoader:
    """
    Class to manage loading and prediction using a machine learning model.

    Attributes:
        _model (object): The machine learning model.
        _dict_vectorizer (object): A dictionary vectorizer for feature transformation.
        _metadata_model (object): Metadata associated with the model.

    Methods:
        reload(): Reloads the model and vectorizer from S3.
        predict(df_values: pd.DataFrame) -> list: Predicts outputs using the loaded model.
        get_metadata(): Retrieves metadata from the model.
    """

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        """
        Initializes the ModelLoader class with no model or vectorizer loaded.
        """
        self._model = None
        self._dict_vectorizer = None
        self._metadata_model = None

    def reload(self):
        """
        Load the model and vectorizer from S3 and save them locally.

        Returns:
            bool: True if the model and vectorizer are successfully loaded, False otherwise.
        """
        op_model = False
        model_local_path = f"{TEMP_MODEL_DIR}/model.pkl"
        try:
            print(MODEL_S3_PATH)
            s3_client.download_file(S3_BUCKET_NAME, MODEL_S3_PATH, model_local_path)
            self._model = pickle.load(open(model_local_path, 'rb'))
            op_model = True
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)
            if os.path.isfile(model_local_path):
                self._model = pickle.load(open(model_local_path, 'rb'))
                op_model = True

        if not op_model:
            return False

        op_dv = False
        dv_local_path = f"{TEMP_MODEL_DIR}/dict_vectorizer.pkl"
        try:
            s3_client.download_file(S3_BUCKET_NAME, DV_S3_PATH, dv_local_path)
            self._dict_vectorizer = pickle.load(open(dv_local_path, 'rb'))
            op_dv = True
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)
            if os.path.isfile(dv_local_path):
                self._dict_vectorizer = pickle.load(open(dv_local_path, 'rb'))
                op_dv = True

        return op_model and op_dv


    def predict(self, df_values: pd.DataFrame) -> list:
        """
        Predict outputs using the loaded model.

        Args:
            df_values (pd.DataFrame): DataFrame containing features for prediction.

        Returns:
            list: Predictions from the model or None if prediction fails.
        """
        if self._dict_vectorizer is None:
            return None

        try:
            x_values = self._dict_vectorizer.transform(df_values.to_dict(orient='records'))
            x_values = pd.DataFrame.sparse.from_spmatrix(x_values).sparse.to_dense()
            columns = dict( (item, str(i)) for i, item in enumerate(x_values.columns))
            x_values = x_values.rename(columns=columns)
            # print(type(x_values))
            # print(x_values.shape)
        except Exception as  e: # pylint: disable=broad-exception-caught
            print(e)

        if self._model is None:
            return None

        try:
            model_out = self._model.predict(x_values)
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)
            return None
        return model_out.tolist()

    def get_metadata(self):
        """
        Retrieve metadata from the model.

        Returns:
            object: Metadata associated with the model.
        """
        return self._metadata_model

def prepare_features(ride):
    """
    Prepare features for prediction from ride data.

    Args:
        ride (dict or list of dict): Dictionary or list of dictionaries containing ride data.

    Returns:
        pd.DataFrame: DataFrame containing transformed features for the model.
    """
    if isinstance(ride, dict):
        ride = [ ride ]
    df = pd.DataFrame.from_records(ride)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_weekday"] = df["pickup_datetime"].apply(lambda x: str(x.weekday()))
    df["pickup_minutes"] = df["pickup_datetime"].apply(lambda x: x.hour + 60 * x.minute)
    df["PULocationID"] = "PU_" + df["PULocationID"].astype(str)
    df["DOLocationID"] = "DO_" + df["DOLocationID"].astype(str)

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

# Instantiate the ModelLoader class
model_loader = ModelLoader()
# Instantiate the Flask app
app = Flask(__name__)

@app.route('/reload', methods=['POST'])
def reload_endpoint():
    """
    Reload the model and vectorizer via an API call.

    Returns:
        Response: JSON response indicating success or failure of the reload operation.
    """
    flag = model_loader.reload()
    if flag:
        return jsonify({"result": "success", "reloaded": True})
    return jsonify({"result": "failed", "reloaded": False})

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Predict outcomes using the loaded model via an API call.

    Returns:
        Response: JSON response containing the predictions.
    """
    ride = request.get_json()["data"]
    features = prepare_features(ride)
    pred = model_loader.predict(features)
    result = "failed"
    if pred is not None:
        result = "success"
    return jsonify({"result": result, "predictions": pred})

@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint to verify the service is running.

    Returns:
        str: Welcome message indicating the service is running.
    """
    return "Zoomcamp application"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
