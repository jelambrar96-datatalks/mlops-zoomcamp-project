# Use the official Ubuntu image as the base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# localstack credentials
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# endpoint of localstack
ENV S3_ENDPOINT_URL=$S3_ENDPOINT_URL
ENV S3_BUCKET_NAME=$S3_BUCKET_NAME

# path os data
ENV MODEL_S3_PATH=$MODEL_S3_PATH
ENV METADATA_S3_PATH=$METADATA_S3_PATH
ENV DV_S3_PATH=$DV_S3_PATH


# Install Python and pip
RUN apt-get update
RUN apt-get install -y python3 python3-pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# install all python dependences
RUN pip install -r requirements.txt

# Expose the port that Gunicorn will listen on
EXPOSE 8000

# Define the command to run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
