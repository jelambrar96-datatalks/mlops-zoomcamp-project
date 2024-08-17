# Start from the Apache Airflow image
FROM apache/airflow:2.9.3

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to start the Airflow webserver
CMD ["airflow", "webserver"]
