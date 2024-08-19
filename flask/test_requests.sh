#!/usr/bin/env bash

cd "$(dirname "$0")"

# docker compose build
docker compose up -d

# sleep 5
# source .env_test
# pipenv install -r requirements.txt
# pipenv run python test_requests.py
# 
# ERROR_CODE=$?
# 
# if [ ${ERROR_CODE} != 0 ]; then
#     docker-compose logs
#     docker-compose down
#     exit ${ERROR_CODE}
# fi

CONTAINER_NAME="test" # taken from docker compose
docker wait $CONTAINER_NAME

# Get the container's exit code
EXIT_CODE=$(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')

# Check if the exit code is different from zero
if [ "$EXIT_CODE" -ne 0 ]; then
  echo "The container $CONTAINER_NAME finished with errors. Exit code: $EXIT_CODE"
  echo "Container logs:"
  docker logs $CONTAINER_NAME
else
    echo "The container $CONTAINER_NAME ran successfully."
fi

docker compose down
