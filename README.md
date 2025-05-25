# Water Management AI System

An intelligent water consumption analysis system for rural water boards,
integrating rainfall data with consumption patterns to detect anomalies and
predict usage.

## Features

- **Data Integration**: Temporal fusion of rainfall and water consumption data
- **Anomaly Detection**: ML-powered detection of suspicious meter readings
- **Consumption Prediction**: Forecast water usage with climate variables
- **Interactive Dashboard**: Real-time visualizations and insights
- **MLflow Integration**: Model versioning and experiment tracking

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lapillaga/junta-analytics.git
cd junta-analytics
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Setup environment variables:

```bash
cp .env.example .env

# Edit .env with your configuration
```

5. Setup database:

```bash
psql -U your_user -d your_db -f database_setup.sql
```

## Usage
- Run Jupyter Lab
```bash
jupyter lab
```

- Start Flask Application
```bash
python src/app.py
```

- Start MLflow UI
```bash
mlflow server --backend-store-uri ./mlflow_runs --host 0.0.0.0 --port 9090
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 9090
```

- Project Structure

  - notebooks/: Jupyter notebooks for data analysis and model training
  - src/: Source code for Flask application and ML models
  - data/: Data storage (raw, processed, models)
  - static/: Static files for web interface

- Technologies Used
  - Data Science: Pandas, NumPy, Scikit-learn
  - Visualization: Plotly, Matplotlib, Seaborn
  - Web Framework: Flask
  - Database: PostgreSQL, SQLAlchemy
  - ML Tracking: MLflow
  - Frontend: HTML5, JavaScript, Bootstrap
