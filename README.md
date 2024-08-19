# Predicting Taxi Trip Duration in New York City Using Machine Learning and MLOps

## 1. Problem Statement

Accurately predicting the duration of taxi trips in New York City is a complex task due to the city's dynamic traffic conditions, diverse weather patterns, and varying demand levels. Traditional methods of estimation often fall short in providing reliable predictions, leading to inefficiencies for both passengers and taxi services. A robust solution is needed to handle the multifaceted nature of these trips and provide accurate duration predictions.

### 1.1 Proposed Solution: Machine Learning Models

Machine learning (ML) models offer a powerful solution to this problem. By leveraging historical trip data and other relevant factors, these models can learn to predict the duration of future trips with a higher degree of accuracy. The ability to incorporate various features such as pickup and drop-off locations, time of day, day of the week, weather conditions, and traffic data makes ML models particularly well-suited for this task.

### 1.2 Available Dataset: TLC Trip Record Data

To develop and train these machine learning models, we can use the **TLC Trip Record Data**. This dataset is provided by the New York City Taxi and Limousine Commission (TLC) and includes detailed records of taxi trips in the city. The dataset contains features such as pickup and drop-off dates and times, locations, trip distances, and fare amounts, among others. This rich dataset provides a comprehensive foundation for training models to predict trip durations accurately.

### 1.3 Importance of MLOps in Managing Model Lifecycle

Implementing MLOps (Machine Learning Operations) is crucial in managing the lifecycle of these machine learning models. MLOps practices help automate and streamline the processes of deploying, monitoring, and updating models. Given the ever-changing nature of traffic patterns, road conditions, and other external factors, models must be regularly updated to maintain their accuracy. MLOps ensures that these updates can be made efficiently and reliably, minimizing the risk of deploying outdated models.

### 1.4 Advantages of Applying Machine Learning and MLOps

Applying machine learning to predict taxi trip durations offers several advantages, including improved accuracy in predictions, enhanced user satisfaction, and optimized fleet management for taxi services. Additionally, the application of MLOps provides significant benefits by keeping these models up to date. Without MLOps, models risk becoming obsolete, leading to decreased accuracy and potentially negative impacts on service quality. With MLOps, continuous integration and deployment processes ensure that models are consistently retrained and fine-tuned in response to new data, preserving their effectiveness over time.

Leveraging machine learning models and adopting MLOps practices is key to addressing the challenge of predicting taxi trip durations in New York City. This approach not only enhances the precision of predictions but also ensures the sustainability and relevance of the models in a dynamic environment.

_____________________________

## 2. Reproducibility:

### 2.1 Instructions to Start the Project

1. **Clone the Project Repository:**
   If you haven't already, clone the project repository to your local machine:
   ```bash
   git clone https://github.com/jelambrar96-datatalks/mlops-zoomcamp-project
   cd mlops-zoomcamp-project
   ```

2. **Set Up the Environment:**
   Ensure the `.env` file is in the project root directory. This file contains all the necessary environment variables. If it’s not already created, create it and copy the content provided above into the file.

   This is a example of `.env` file.

```bash
### ENV VARIABLES
PROJECT_NAME="mlops-zoomcamp"

### FOR LOCALSTACK
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-east-1

S3_BUCKET_NAME="${PROJECT_NAME}-bucket"

### FOR AIRFLOW
AIRFLOW_IMAGE_NAME="apache/airflow:2.9.3"
AIRFLOW_UID=1000
AIRFLOW_PROJ_DIR="./airflow"
_AIRFLOW_WWW_USER_USERNAME="airflow"
_AIRFLOW_WWW_USER_PASSWORD="airflow"
_PIP_ADDITIONAL_REQUIREMENTS="boto3==1.34.131 localstack==3.6.0 mlflow==2.15.1 numpy==1.26.4 pandas==2.1.4 pyarrow==15.0.2 requests==2.32.3 scikit-learn==1.5.1 s3fs==2024.6.1"

AIRFLOW_START_TIME="2023-01-01"

# MLFLOW
MLFLOW_PORT=5001
MLFLOW_POSTGRES_USER=mlflow
MLFLOW_POSTGRES_PASS=mlflow
MLFLOW_BUCKET=mlflow-bucket

# GRAFANA
GRAFANA_PORT=3000
GRAFANA_POSTGRES_USER=grafana
GRAFANA_POSTGRES_PASS=grafana
``` 

3. **Build the Docker Images:**
   Some services require building Docker images from custom Dockerfiles. Use the following command to build those images:
   ```bash
   docker-compose build
   ```

4. **Start the Docker Containers:**
   To start all the services defined in the `docker-compose.yml` file, run:
   ```bash
   docker-compose up -d
   ```
   The `-d` flag runs the containers in detached mode, meaning they will run in the background.

5. **Verify All Services Are Running:**
   Use the following command to list all running containers and verify that everything started correctly:
   ```bash
   docker-compose ps
   ```

### 2.2 Available Applications and Ports on Host

Once the project is running, the following applications will be available on your host:

1. **Airflow Webserver:**
   - **URL:** `http://localhost:8080`
   - **Port:** 8080

2. **Flower (Celery Monitoring Tool):**
   - **URL:** `http://localhost:5555`
   - **Port:** 5555

3. **Mlflow Tracking Server:**
   - **URL:** `http://localhost:5001`
   - **Port:** 5001

4. **Grafana Dashboard:**
   - **URL:** `http://localhost:3000`
   - **Port:** 3000

5. **Flask Application:**
   - **URL:** `http://localhost:8000`
   - **Port:** 8000

### 2.3 Notes

- Ensure Docker and Docker Compose are installed on your machine before running these commands.
- If any service fails to start, check the logs using `docker-compose logs <service_name>`.
- You can stop all services by running `docker-compose down`, which will also remove the containers.

_____________________________

## 3 NO-cloud used, but `localstack` and `docker-compose` impleplemented

To address the problem of predicting taxi trip durations through MLOps, a decision was made to implement a solution that runs locally rather than in the cloud. This local deployment approach offers several key advantages:

### 3.1 Advantages of Local Development

1. **Cost Efficiency**: Running the solution locally eliminates the need for cloud infrastructure costs, such as pay-per-use charges and data transfer fees. This can result in significant savings, especially during the development and testing phases.

2. **Improved Control and Security**: Local deployment provides greater control over the environment, allowing for more stringent security measures and easier access management. Developers can have direct oversight of the entire infrastructure, reducing potential vulnerabilities associated with cloud environments.

3. **Faster Iteration**: Working locally can lead to faster iteration cycles, as developers don't need to rely on internet connectivity or deal with cloud service latency. This speed can be particularly beneficial during the initial stages of model development and testing.

4. **Custom Environment Configuration**: A local setup allows for highly customized configurations tailored specifically to the project’s needs. Developers can tweak settings and optimize performance without being constrained by the limitations or default configurations of cloud services.

### 3.2 Containerized Infrastructure with Docker and Docker Compose

Despite being a local solution, Docker and Docker Compose were used to build a container-based infrastructure. This approach enables the entire environment to be packaged into containers, making it portable and consistent across different development machines. The use of Docker ensures that the solution can be easily scaled to the cloud if needed in the future. Docker Compose simplifies the orchestration of these containers, allowing for a streamlined and efficient setup of complex multi-container applications.

### 3.3 LocalStack for Simulated Cloud Storage

Although the solution does not utilize the cloud, LocalStack was implemented to simulate cloud services for data storage. LocalStack mimics the behavior of AWS S3 buckets, allowing developers to work with cloud-like storage services directly from their local environment. This ensures that the transition to a full cloud deployment can be seamless, as the local development environment closely mirrors the cloud infrastructure.

_______________________________________________________________________________

## 4. Experiment tracking and model registry with `mlflow`

