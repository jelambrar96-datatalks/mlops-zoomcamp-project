#!/bin/bash

aws s3 mb s3://${S3_BUCKET_NAME} --endpoint-url http://localstack:4566
aws s3 mb s3://${MLFLOW_BUCKET} --endpoint-url http://localstack:4566

aws --endpoint-url=http://localstack:4566 iam create-role --role-name zoomcamp-role --assume-role-policy-document file://trust-policy.json

envsubst < ./policy.template.json > ./policy.json
aws --endpoint-url=http://localstack:4566 iam put-role-policy --role-name zoomcamp-role --policy-name zoomcamp-policy --policy-document file://policy.json
