# 🚀 MLB Pitcher Velocity Decline Predictor

> Early warning system for pitcher fatigue using Statcast data

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)]()
[![MLflow](https://img.shields.io/badge/MLflow-2.1.1-orange)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)]()

Predicts probability of >1.5mph fastball velocity drop in next 30 days using biomechanical and workload metrics.

![Prediction Dashboard](docs/dashboard_screenshot.png)

## Features
- **Data Pipeline**: Automated Statcast data ingestion
- **MLflow Tracking**: Experiment logging & model registry
- **FastAPI**: Production inference endpoints
- **SHAP Explanations**: Model interpretability
- **Drift Detection**: Automated data shift monitoring

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/mlb-velocity-decline-predictor.git
cd mlb-velocity-decline-predictor

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python src/models/train.py

# Start API locally
uvicorn src.app.api:app --reload
```

## Data Sources
- [Statcast Search](https://baseballsavant.mlb.com/statcast_search)
- [Lahman Database](http://www.seanlahman.com/baseball-archive/statistics/)
- [Weather Data Integration](docs/weather_integration.md)

## Deployment
```bash
# Build and deploy with Docker
docker compose up --build
```

Access services:
- API: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Dashboard: `http://localhost:8501`

## Project Structure
```mermaid
graph TD
    A[mlb-velocity-decline-predictor] --> B[.github/workflows]
    A --> C[data]
    C --> C1[raw]
    C --> C2[processed]
    A --> D[deployment]
    A --> E[models]
    E --> E1[production]
    A --> F[src]
    F --> F1[app]
    F1 --> F11[api.py]
    F1 --> F12[schemas.py]
    F --> F2[models]
    F2 --> F21[train.py]
    F --> F3[utils]
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
