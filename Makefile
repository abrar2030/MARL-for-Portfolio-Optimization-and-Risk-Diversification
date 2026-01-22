.PHONY: help install test train eval clean docker-build docker-up docker-down benchmark analysis

help:
	@echo "MARL Portfolio Optimization - Production System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make test            - Run tests with coverage"
	@echo "  make train           - Train full model"
	@echo "  make train-lite      - Train MARL-Lite model"
	@echo "  make eval            - Evaluate trained model"
	@echo "  make benchmark       - Run performance benchmarks"
	@echo "  make analysis        - Run feature importance analysis"
	@echo "  make rebalancing     - Run rebalancing optimization"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-up       - Start production stack"
	@echo "  make docker-down     - Stop all services"
	@echo "  make api             - Start API server"
	@echo "  make dashboard       - Start dashboard"
	@echo "  make clean           - Clean generated files"

install:
	pip install -r requirements.txt -r requirements-prod.txt
	pip install -e .

test:
	pytest tests/ -v --cov=code --cov-report=html --cov-report=term

test-quick:
	pytest tests/test_comprehensive.py::TestIntegration -v

train:
	python code/main.py --mode train --episodes 300 --save-dir ./results/training

train-lite:
	python code/main.py --mode train --config configs/marl_lite.json --episodes 200 --save-dir ./results/lite_training

eval:
	python code/main.py --mode eval --load-model ./results/training/best_model

benchmark:
	python code/benchmarks/run_benchmarks.py

analysis:
	python code/analysis/feature_importance.py

rebalancing:
	python code/analysis/rebalancing_optimization.py

docker-build:
	docker-compose build

docker-up:
	docker-compose --profile production up -d

docker-down:
	docker-compose --profile production down

docker-logs:
	docker-compose logs -f

docker-test:
	docker-compose --profile test up

docker-benchmark:
	docker-compose --profile benchmark up

api:
	uvicorn code.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	python code/dashboard/app.py --port 8050

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/

lint:
	flake8 code/ --max-line-length=100 --ignore=E203,W503

format:
	black code/ tests/

check:
	make lint
	make test

all: clean install test train

.DEFAULT_GOAL := help
