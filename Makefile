# Project shortcuts
init:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/unit
	pytest tests/integration

train:
	python src/models/train.py

run-api:
	uvicorn src.app.api:app --reload

monitor-drift:
	python src/monitoring/drift.py

docker-build:
	docker build -t velocity-predictor:latest -f deployment/Dockerfile .

docker-run:
	docker run -p 8000:8000 velocity-predictor
