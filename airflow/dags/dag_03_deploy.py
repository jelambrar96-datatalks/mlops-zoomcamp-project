"""
Airflow DAG that compared all registred mlflow
models and deploy the better
"""


import os
import json

from io import BytesIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import boto3
import pandas as pd
import requests
import requests.exceptions as rexcep
import s3fs # pylint: disable=unused-import

import pickle

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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


dag_03_deploy = DAG(
    'dag_03_deploy',
    default_args=default_args,
    description='A simple DAG to compare models and deployed',
    schedule_interval='0 2 1 * *', # At 00:00 on day-of-month 1.
    start_date=datetime.strptime(AIRFLOW_START_TIME, "%Y-%m-%d"),
    catchup=False,
)


task_start = DummyOperator(
    task_id="task_start",
    dag=dag_03_deploy
)


def function_download_model(download_date: str):
    """
    1. donwload best model from mlflow. according rmse
    2. upload model on s3
    """

    # Configura la URI de seguimiento de MLflow si es necesario
    mlflow.set_tracking_uri("http://mlflow:5000")  # from docker-compose

    # Especifica el nombre del experimento
    experiment_name = "mlops-zoomcamp-experiment"
    client = MlflowClient()

    # Obtén el ID del experimento
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Lista las ejecuciones del experimento, ordenadas por RMSE
    runs = client.search_runs(experiment_ids=[experiment_id],
                            filter_string="",
                            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                            order_by=["metrics.rmse ASC"])

    # Obtén la primera ejecución, que debería ser la de mejor RMSE
    best_run = runs[0]
    best_rmse = best_run.data.metrics["rmse"]
    best_run_id = best_run.info.run_id
    print(f"El mejor RMSE es: {best_rmse}, en la ejecución {best_run_id}")

    # Cargar el modelo desde la URI correspondiente
    # model_uri = f"runs:/{best_run_id}/model.pkl"
    # model = mlflow.pyfunc.load_model(model_uri)

    best_run_uri = f"{best_run.info.artifact_uri}/{best_run.data.params['model_name']}"
    model = mlflow.pyfunc.load_model(best_run_uri)

    os.makedirs("/tmp/mlrun/best_model/", exist_ok=True)
    model_path = "/tmp/mlrun/best_model/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    model_info = {
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": best_run.info.run_id,
        "rmse": best_run.data.metrics["rmse"],
        "model_uri": f"runs:/{best_run.info.run_id}/model",
        "params": best_run.data.params,
        "metrics": best_run.data.metrics,
        "tags": best_run.data.tags,
        "execution_date": download_date
    }

    model_metadata_path = "/tmp/mlrun/best_model/model_metadata.json"
    with open(model_metadata_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_info, indent=4, sort_keys=True))

    # load model on localstack
    s3_dir_path = "models/skmodels/best_model"
    s3_model_path = f"{s3_dir_path}/model.pkl"
    s3_client.upload_file(model_path, S3_BUCKET_NAME, s3_model_path)
    # load metadata
    # s3_metadata_model_path = f"{s3_dir_path}/model_metadata.json"
    # s3_client.upload_file(model_metadata_path, S3_BUCKET_NAME, s3_metadata_model_path)
    # load dict vectorizer
    local_dv_path = f"{best_run.data.params['local_model_path']}/dict_vectorizer.pkl"
    s3_dv_path = f"{s3_dir_path}/dict_vectorizer.pkl"
    s3_client.upload_file(local_dv_path, S3_BUCKET_NAME, s3_dv_path)





task_download_model = PythonOperator(
    task_id='task_download_model',
    python_callable=function_download_model,
    dag=dag_03_deploy,
    op_kwargs={
        "download_date": "{{ ds }}",
    }
)


end_start = DummyOperator(
    task_id="end_start",
    dag=dag_03_deploy
)


task_start >> task_download_model >> end_start # pylint: disable=pointless-statement
