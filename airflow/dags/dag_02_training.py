"""
train and eval model
"""

from io import BytesIO
import os

import pickle
import logging

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


import s3fs # pylint: disable=unused-import
import boto3


import pyarrow # pylint: disable=unused-import
import pyarrow.parquet as pq
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import (
    LinearRegression,
    Lasso
)
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from airflow import DAG
from airflow.operators.dummy import DummyOperator # pylint: disable=import-error,no-name-in-module
from airflow.operators.python import PythonOperator



# -----------------------------------------------------------------------------

# get env vars
AIRFLOW_START_TIME = os.getenv("AIRFLOW_START_TIME", "2023-01-01")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

# mlflow.set_tracking_uri("http://mlflow:5001") #TODO: using env vars
# mlflow.set_experiment("mlops-zoomcamp") #TODO: using env vars

# -----------------------------------------------------------------------------

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


dag_02_training = DAG(
    dag_id="dag_02_training",
    default_args=default_args,
    description='A simple DAG to train model, registrer on mlflow and store in s3',
    schedule_interval='0 1 1 * *', # At 01:00 on day-of-month 1.
    start_date=datetime.strptime(AIRFLOW_START_TIME, "%Y-%m-%d"),
    catchup=False,
)

# -----------------------------------------------------------------------------

task_start = DummyOperator(
    task_id="task_start",
    dag=dag_02_training
)

# -----------------------------------------------------------------------------

def read_parquet_from_s3(s3, bucket, key):
    """
    Download s3 file
    """
    # Descargar el archivo parquet desde S3
    response = s3.get_object(Bucket=bucket, Key=key)
    # Leer el archivo parquet en un DataFrame de pandas
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
        sample=None) -> pd.DataFrame:
    """
    concatenate all parquet files
    """
    output_df = None
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        print(current_datetime)
        current_year = current_datetime.year  # Cambia al año correspondiente
        current_month = current_datetime.month  # Cambia al mes correspondiente
        delta = relativedelta(year=current_year, month=current_month, months=1)
        current_datetime = current_datetime + delta
        key_file = f"nyc-taxi-data/type={type_tripdata}/year={current_year:04d}/month={current_month:02d}/raw-data.parquet" # pylint: disable=line-too-long
        try:
            current_df = read_parquet_from_s3(bucket=bucket_name, key=key_file, s3=s3_object)
        except: # pylint: disable=bare-except
            continue
        if sample is not None and current_df.shape[0] > sample:
            current_df = current_df.sample(sample, random_state=1) # reduce size of dataframe
        if output_df is None:
            output_df = current_df.copy()
            continue
        output_df = pd.concat((output_df, current_df))
    if sample is not None and output_df.shape[0] > sample:
        output_df = output_df.sample(sample, random_state=1)
    return output_df


def get_base_pattern_dataset_file(download_date: datetime) -> str :
    """
    get_base_pattern_dataset_file
    """
    path_to_save = "nyc-taxi-data/datasets/year={year:04d}/month={month:02d}" # pylint: disable=line-too-long
    path_to_save = path_to_save.format(
        year=download_date.year,
        month=download_date.month,
    )
    return path_to_save


def get_patern_dataset_file(file_basename: str, download_date: datetime) -> str :
    """
    store data
    """
    base_path = get_base_pattern_dataset_file(download_date=download_date)
    path_to_save = "s3://{s3_bucket_name}/{base_path}/{basename}.parquet" # pylint: disable=line-too-long
    path_to_save = path_to_save.format(
        s3_bucket_name=S3_BUCKET_NAME,
        base_path=base_path,
        basename=file_basename
    )
    return path_to_save


def dataframe_to_s3(
        df: pd.DataFrame,
        file_basename: str,
        download_date: datetime
        ) -> None:
    """
    store dataframe as parquet on s3
    """
    path_to_save = get_patern_dataset_file(file_basename, download_date)
    df.to_parquet(
        path_to_save,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=STORAGE_OPTIONS,
    )


def dump_pickle(obj, filename: str) -> None:
    """
    store object using pickle
    """
    try:
        with open(filename, "wb") as f_out:
            pickle.dump(obj, f_out)
    finally:
        pass


def csr_to_df(x_csr, y_csr):
    """
    # Ejemplo de conversión de csr_matrix a DataFrame
    """
    x_df = pd.DataFrame.sparse.from_spmatrix(x_csr).sparse.to_dense()
    # num_columns = x_df.shape[1]
    # columns_names = dict( (i, f"col_{i}") for i in range(num_columns) )
    # _df = x_df.rename(columns=columns_names)
    # Asumimos que Y_csr es también csr_matrix
    y_df = pd.DataFrame(y_csr, columns=['target'])
    return x_df, y_df


def create_dataset(
        download_date,
        type_tripdata,
        start_date,
        n_sample):
    """
    capture dataset from s3 localstack
    """
    download_date = datetime.strptime(download_date, "%Y-%m-%d")
    df = get_parquet_files(
        type_tripdata=type_tripdata,
        start_datetime=datetime.strptime(start_date, "%Y-%m-%d"),
        end_datetime=download_date,
        bucket_name=S3_BUCKET_NAME,
        s3_object=s3_client,
        sample=n_sample)

    df["pickup_weekday"] = df["pickup_datetime"].apply(lambda x: str(x.weekday))
    df["pickup_minutes"] = df["pickup_datetime"].apply(lambda x: x.hour + 60 * x.minute)
    df["PULocationID"] = "PU_" + df["PULocationID"]
    df["DOLocationID"] = "DO_" + df["DOLocationID"]

    drop_columns = [
        "payment_type",
        "tip_amount",
        "tolls_amount",
        "total_amount",
        "store_and_fwd_flag",
        "passenger_count",
        "mta_tax",
        "improvement_surcharge",
        "fare_amount",
        "extra",
        "congestion_surcharge",
        "VendorID",
        "RatecodeID",
        "payment_type",
        "pickup_datetime",
        "dropoff_datetime",
    ]

    df = df.drop(columns=drop_columns)

    dv = DictVectorizer()

    numerical_cols = [
        "pickup_minutes",
        "trip_distance"
    ]

    categorical_cols = [
        "PULocationID",
        "DOLocationID",
        "pickup_weekday",
        # "PU_DO_location",
    ]

    target = 'duration'

    x_data = dv.fit_transform(df[categorical_cols + numerical_cols].to_dict(orient='records'))
    y_label = df[target].values
    df = pd.concat(csr_to_df(x_csr=x_data, y_csr=y_label), axis=1)

    ind_60 = int(df.shape[0] * 0.6)
    ind_80 = int(df.shape[0] * 0.8)
    df = shuffle(df)
    df_train = df[:ind_60]
    df_val   = df[ind_60:ind_80]
    df_test  = df[ind_80:]

    # logging.info(", ".join(df.columns))
    dv_path = "/tmp/dict_vectorizer.pkl"
    dump_pickle(dv, dv_path)
    dv_s3_path = get_base_pattern_dataset_file(download_date=download_date)
    s3_client.upload_file(dv_path, S3_BUCKET_NAME, f"{dv_s3_path}/dict_vectorizer.pkl")

    if len(list(df_train.columns)) != len(set(df_train.columns)):
        columns_str = ", ".join(df_train.columns)
        raise ValueError(f"Duplicated columns: {columns_str}")
    dataframe_to_s3(
        df=df_train,
        file_basename="train",
        download_date=download_date
    )
    if len(list(df_test.columns)) != len(set(df_test.columns)):
        columns_str = ", ".join(df_test.columns)
        raise ValueError(f"Duplicated columns: {columns_str}")
    dataframe_to_s3(
        df=df_test,
        file_basename="test",
        download_date=download_date
    )
    if len(list(df_val.columns)) != len(set(df_val.columns)):
        columns_str = ", ".join(df_val.columns)
        raise ValueError(f"Duplicated columns: {columns_str}")
    dataframe_to_s3(
        df=df_val,
        file_basename="val",
        download_date=download_date
    )


task_create_dataset = PythonOperator(
    task_id="task_create_dataset",
    dag=dag_02_training,
    python_callable=create_dataset,
    op_kwargs={
    "download_date": "{{ ds }}",
    "type_tripdata": "yellow",
    "start_date": AIRFLOW_START_TIME,
    "n_sample": 10000
    }
)


# -----------------------------------------------------------------------------

task_trigger_sklearn_models = DummyOperator(
    task_id="task_trigger_sklearn_models",
    dag=dag_02_training
)

# -----------------------------------------------------------------------------


def split_dataset_data(df: pd.DataFrame, target="duration"):
    """
    split_dataset_data
    """
    y_data = df[[target]]
    x_data = df.drop(columns=[target])
    return x_data, y_data


def eval_metrics(actual, pred):
    """compute metrics"""
    mse = mean_squared_error(actual, pred)
    rmse = root_mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, rmse, mae, r2


def get_read_patern_dataset_file(file_basename: str, download_date: datetime) -> str :
    """
    store data
    """
    path_to_save = "nyc-taxi-data/datasets/year={year:04d}/month={month:02d}/{basename}.parquet" # pylint: disable=line-too-long
    path_to_save = path_to_save.format(
        year=download_date.year,
        month=download_date.month,
        basename=file_basename
    )
    return path_to_save


def train_sklearn_model(download_date, sklearn_model, model_name):
    """
    train sklearn model
    """
    # artifact_path = f'{model_name}-{download_date}'
    artifact_path = f'{model_name}'
    download_date = datetime.strptime(download_date, "%Y-%m-%d")
    # load data from localstack
    key_train = get_read_patern_dataset_file(file_basename="train", download_date=download_date)
    key_test  = get_read_patern_dataset_file(file_basename="test", download_date=download_date)
    key_val   = get_read_patern_dataset_file(file_basename="val", download_date=download_date)
    base_key_dv = get_base_pattern_dataset_file(download_date=download_date)
    key_dv = f"{base_key_dv}/dict_vectorizer.pkl"

    print(key_train)
    logging.info(key_train)

    # create dict vectorizer
    x_train, y_train = split_dataset_data(
        df=read_parquet_from_s3(bucket=S3_BUCKET_NAME, key=key_train, s3=s3_client),
        target="target"
    )
    x_test, y_test = split_dataset_data(
        df=read_parquet_from_s3(bucket=S3_BUCKET_NAME, key=key_test, s3=s3_client),
        target="target"
    )
    x_val, y_val = split_dataset_data(
        df=read_parquet_from_s3(bucket=S3_BUCKET_NAME, key=key_val, s3=s3_client),
        target="target"
    )

    experiment_name = "mlops-zoomcamp-experiment"
    mlflow.set_tracking_uri("http://mlflow:5000") # taken from docker compose
    mlflow.set_experiment(experiment_name=experiment_name)

    local_artifacts_path = "/tmp/artifacts"
    os.makedirs(local_artifacts_path, exist_ok=True)

    # with mlflow.start_run(
    #     experiment_id="mlops-zoomcamp"
    # ):
    with mlflow.start_run() as run:

        sklearn_model.fit(x_train, y_train)

        y_pred_1 = sklearn_model.predict(x_test)
        signature_1 = infer_signature(x_test, y_pred_1)

        mlflow.log_params(sklearn_model.get_params())
        mlflow.log_param("train-data-path", key_train)
        mlflow.log_param("test-data-path",  key_test)
        mlflow.log_param("valid-data-path", key_val)
        mlflow.log_param("dict-vectorizer-path", key_dv)

        mse, rmse, mae, r2 = eval_metrics(y_test, y_pred_1)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse",  mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # final_name = datetime.now().strftime("%Y%m%dT%H%M%S")
        # artifact_path = os.path.join(
        # temp_artifact_dir, f"models/sklearn-model-artifacts-{final_name}")
        # os.makedirs(artifact_path, exist_ok=True)

        # mlflow.sklearn.log_artifact(artifact_path)
        model_info = mlflow.sklearn.log_model(
            sk_model=sklearn_model,
            artifact_path=artifact_path,
            # artifact_path=f"sklearn-model-artifacts-{final_name}",
            signature=signature_1,
            registered_model_name=model_name
        )

        print(model_info)
        logging.info(model_info)

        dv_path = f"{mlflow.get_artifact_uri()}/{model_name}/dict_vectorizer.pkl"
        s3_client.download_file(S3_BUCKET_NAME, key_dv, dv_path)


task_train_linear_regression = PythonOperator(
    task_id="task_train_linear_regression",
    dag=dag_02_training,
    python_callable=train_sklearn_model,
    op_kwargs={
        "download_date": "{{ ds }}",
        "sklearn_model": LinearRegression(),
        "model_name": "sklearn-linear-regression"
    }
)

task_train_lasso = PythonOperator(
    task_id="task_train_lasso",
    dag=dag_02_training,
    python_callable=train_sklearn_model,
    op_kwargs={
        "download_date": "{{ ds }}",
        "sklearn_model": Lasso(),
        "model_name": "sklearn-lasso"
    }
)

task_train_gradient_boost = PythonOperator(
    task_id="task_train_gradient_boost",
    dag=dag_02_training,
    python_callable=train_sklearn_model,
    op_kwargs={
        "download_date": "{{ ds }}",
        "sklearn_model": GradientBoostingRegressor(),
        "model_name": "sklearn-gradient-boosting-regression"
    }
)

task_train_random_forest = PythonOperator(
    task_id="task_train_random_forest",
    dag=dag_02_training,
    python_callable=train_sklearn_model,
    op_kwargs={
        "download_date": "{{ ds }}",
        "sklearn_model": RandomForestRegressor(),
        "model_name": "sklearn-random-forest-regression"
    }
)

# -----------------------------------------------------------------------------

task_end_sklearn_models = DummyOperator(
    task_id="task_end_sklearn_models",
    dag=dag_02_training
)

# pylint: disable=pointless-statement
task_start >> task_create_dataset >> task_trigger_sklearn_models
task_trigger_sklearn_models >> task_train_linear_regression >> task_end_sklearn_models
task_trigger_sklearn_models >> task_train_lasso             >> task_end_sklearn_models
task_trigger_sklearn_models >> task_train_gradient_boost    >> task_end_sklearn_models
task_trigger_sklearn_models >> task_train_random_forest     >> task_end_sklearn_models

# pylint: enable=pointless-statement
