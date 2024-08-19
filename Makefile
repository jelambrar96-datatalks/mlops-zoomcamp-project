LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")

test:
	pytest flask/test_app.py

quality_checks:
	isort flask/*.py
	isort airflow/dags/*.py
	black flask/*.py
	black airflow/dags/*.py
	pylint flask/*.py
	pylint airflow/dags/*.py

build: quality_checks test
	docker compose --env-file .env build

integration_test: build
	bash flask/test_requests.sh

setup:
	pre-commit install
