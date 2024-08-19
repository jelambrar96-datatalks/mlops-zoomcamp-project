# Predicting Taxi Trip Duration in New York City Using Machine Learning and MLOps

## 1. Problem Statement

Accurately predicting the duration of taxi trips in New York City is a complex task due to the city's dynamic traffic conditions, diverse weather patterns, and varying demand levels. Traditional methods of estimation often fall short in providing reliable predictions, leading to inefficiencies for both passengers and taxi services. A robust solution is needed to handle the multifaceted nature of these trips and provide accurate duration predictions.

### 1.1. Proposed Solution: Machine Learning Models

Machine learning (ML) models offer a powerful solution to this problem. By leveraging historical trip data and other relevant factors, these models can learn to predict the duration of future trips with a higher degree of accuracy. The ability to incorporate various features such as pickup and drop-off locations, time of day, day of the week, weather conditions, and traffic data makes ML models particularly well-suited for this task.

### 1.2. Available Dataset: TLC Trip Record Data

To develop and train these machine learning models, we can use the **TLC Trip Record Data**. This dataset is provided by the New York City Taxi and Limousine Commission (TLC) and includes detailed records of taxi trips in the city. The dataset contains features such as pickup and drop-off dates and times, locations, trip distances, and fare amounts, among others. This rich dataset provides a comprehensive foundation for training models to predict trip durations accurately.

### 1.3. Importance of MLOps in Managing Model Lifecycle

Implementing MLOps (Machine Learning Operations) is crucial in managing the lifecycle of these machine learning models. MLOps practices help automate and streamline the processes of deploying, monitoring, and updating models. Given the ever-changing nature of traffic patterns, road conditions, and other external factors, models must be regularly updated to maintain their accuracy. MLOps ensures that these updates can be made efficiently and reliably, minimizing the risk of deploying outdated models.

### 1.4. Advantages of Applying Machine Learning and MLOps

Applying machine learning to predict taxi trip durations offers several advantages, including improved accuracy in predictions, enhanced user satisfaction, and optimized fleet management for taxi services. Additionally, the application of MLOps provides significant benefits by keeping these models up to date. Without MLOps, models risk becoming obsolete, leading to decreased accuracy and potentially negative impacts on service quality. With MLOps, continuous integration and deployment processes ensure that models are consistently retrained and fine-tuned in response to new data, preserving their effectiveness over time.

Leveraging machine learning models and adopting MLOps practices is key to addressing the challenge of predicting taxi trip durations in New York City. This approach not only enhances the precision of predictions but also ensures the sustainability and relevance of the models in a dynamic environment.

_____________________________

## 2. Reproducibility:

### 2.1. Instructions to Start the Project

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

### 2.2. Available Applications and Ports on Host

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

### 2.3. Notes

- Ensure Docker and Docker Compose are installed on your machine before running these commands.
- If any service fails to start, check the logs using `docker-compose logs <service_name>`.
- You can stop all services by running `docker-compose down`, which will also remove the containers.

_____________________________

## 3. NO-cloud used, but `localstack` and `docker-compose` impleplemented

To address the problem of predicting taxi trip durations through MLOps, a decision was made to implement a solution that runs locally rather than in the cloud. This local deployment approach offers several key advantages:

### 3.1. Advantages of Local Development

1. **Cost Efficiency**: Running the solution locally eliminates the need for cloud infrastructure costs, such as pay-per-use charges and data transfer fees. This can result in significant savings, especially during the development and testing phases.

2. **Improved Control and Security**: Local deployment provides greater control over the environment, allowing for more stringent security measures and easier access management. Developers can have direct oversight of the entire infrastructure, reducing potential vulnerabilities associated with cloud environments.

3. **Faster Iteration**: Working locally can lead to faster iteration cycles, as developers don't need to rely on internet connectivity or deal with cloud service latency. This speed can be particularly beneficial during the initial stages of model development and testing.

4. **Custom Environment Configuration**: A local setup allows for highly customized configurations tailored specifically to the project’s needs. Developers can tweak settings and optimize performance without being constrained by the limitations or default configurations of cloud services.

### 3.2. Containerized Infrastructure with Docker and Docker Compose

Despite being a local solution, Docker and Docker Compose were used to build a container-based infrastructure. This approach enables the entire environment to be packaged into containers, making it portable and consistent across different development machines. The use of Docker ensures that the solution can be easily scaled to the cloud if needed in the future. Docker Compose simplifies the orchestration of these containers, allowing for a streamlined and efficient setup of complex multi-container applications.

### 3.3. LocalStack for Simulated Cloud Storage

Although the solution does not utilize the cloud, LocalStack was implemented to simulate cloud services for data storage. LocalStack mimics the behavior of AWS S3 buckets, allowing developers to work with cloud-like storage services directly from their local environment. This ensures that the transition to a full cloud deployment can be seamless, as the local development environment closely mirrors the cloud infrastructure.

_______________________________________________________________________________

## 4. Experiment tracking and model registry with `mlflow`

For experiment tracking and model registry, **MLflow** was utilized throughout the project. The primary goal was to build a machine learning model capable of predicting taxi trip durations based on several key features, including trip distance, pickup and drop-off locations, day of the week, and time of day.

In this process, the following machine learning regression models were registered using MLflow:

1. **Linear Regression**
2. **Lasso Regression**
3. **Gradient Boosting**
4. **Random Forests**

For each model, the following performance metrics were tracked and stored:

- **Root Mean Squared Error (RMSE)**
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

These metrics provided a comprehensive evaluation of the models' accuracy and effectiveness in predicting taxi trip durations, enabling an informed selection of the best-performing model for deployment.
_______________________________________________________________________________

## 5. Workflow orchestration with airflow

For workflow orchestration, **Airflow** was implemented to manage and automate the various processes involved in the project. Airflow offers several advantages in workflow orchestration:

### Advantages of Airflow in Workflow Orchestration

1. **Scalability**: Airflow can handle complex workflows involving large-scale data processing. It allows for the creation of Directed Acyclic Graphs (DAGs) that can be scaled as needed, whether running on a single machine or distributed across a cluster.

2. **Flexibility**: Airflow provides the flexibility to define workflows in Python, making it easy to integrate with existing codebases and customize workflows according to specific project requirements.

3. **Robust Scheduling**: With Airflow, tasks can be scheduled to run at specific times or triggered based on dependencies. This ensures that processes are executed in the correct order and without manual intervention.

4. **Monitoring and Logging**: Airflow includes built-in monitoring and logging features, which allow for real-time tracking of workflow execution, making it easier to identify and resolve issues promptly.

### Implemented DAGs

Three main DAGs were implemented to manage the end-to-end process:

1. **`dag01_download_data`**: 
   - **Purpose**: This DAG is responsible for downloading the NYC taxi dataset, applying necessary filtering processes, and storing the cleaned data files within LocalStack. This step ensures that the data is prepared and available for model training.

2. **`dag02_training`**: 
   - **Purpose**: This DAG is used to build the dataset, train the machine learning models, and register the trained models in MLflow. It automates the entire training pipeline, ensuring consistency and repeatability across model versions.

3. **`dag03_deploy`**: 
   - **Purpose**: This DAG selects the model with the best Root Mean Squared Error (RMSE) from the models registered in MLflow. It then uploads the selected model to an S3 bucket simulated by LocalStack, where it can be accessed by a Flask application for making predictions.

By leveraging Airflow for workflow orchestration, the project benefits from automated, reliable, and scalable management of data processing, model training, and deployment processes. This approach ensures that each step is executed efficiently and in the correct sequence, facilitating the seamless integration of machine learning models into the production environment.

_______________________________________________________________________________

## 6. Model deployment with flask

A Flask API was designed to load the best-performing model stored in S3. This model is then used to predict the duration of taxi trips based on the time of day and the pickup and drop-off locations.

### Advantages of Combining Flask with Machine Learning Applications

One of the key advantages of using Flask is its ease of integration with Python-based machine learning applications. Flask is lightweight, flexible, and allows developers to quickly set up web applications that can interact with machine learning models. This combination enables seamless deployment of predictive models, allowing for real-time inference through a simple and efficient web interface.

### API Functionality

The Flask API is capable of the following:

- **Loading the Best Model**: The API automatically loads the best model, selected based on the lowest RMSE, from the S3 bucket.
- **Predicting Trip Duration**: Given inputs such as the time of day and the starting and ending locations of a taxi trip, the API processes these inputs through the model to predict the expected trip duration.

This setup ensures that the machine learning model can be accessed and utilized in a production environment, providing accurate and timely predictions for end users.

_______________________________________________________________________________

## 7. Model monitoring with grafana, prefect and evidently

_______________________________________________________________________________

## 8. Best practices 

### 8.1 Unit test to model deployment

### 8.2 Integration test to model deployment 

### 8.3 Linter and formater code 


### 8.4 Makefile 


### 8.5 Pre-commit hooks


### CI/CD Pipeline with Jenkins






