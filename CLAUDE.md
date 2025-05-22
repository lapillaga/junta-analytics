# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project: Junta Analytics

## Project Overview

The idea behind this project is to create a web application that provides analytics and insights
related to a Water Quality Monitoring System.
The application purpose is a parish water utility designed to turn raw 
consumption and rainfall data into actionable insights. 
It will consist of:
Data Integration
- Read meter readings from a PostgreSQL database.
- Import dekadal rainfall indicators (CHIRPS CSV) for the same administrative unit.

Processing & Fusion
- Aggregate both sources on a dekadal basis.
- Merge into a single dataset combining consumption and precipitation by date.

Analysis & Modeling
- Perform exploratory analysis (correlations, dry vs. normal period comparisons).
- Train an anomaly detection model (e.g., Isolation Forest) to classify each meter reading as normal or anomalous within expected consumption ranges.
- Persist the trained anomaly detector to disk (e.g., models/anomaly_detector.pkl) for later inference.
- Forecast future demand with a regression model (Random Forest or Prophet) based on past consumption and rainfall indicators.

Visualization & Service
- Generate charts (Matplotlib/Bokeh) showing:
  - Consumption vs. rainfall with anomaly markers.
  - Expected consumption range bands vs. actual readings, highlighting deviations.
- Expose data, predictions, and graphics through a Flask microservice (JSON and PNG endpoints), including an endpoint to predict anomaly status for a single reading.

Problem Addressed
- Lack of timely insights to anticipate demand spikes, detect leaks early, and adjust rates or maintenance proactively—enhancing operational efficiency and minimizing water losses.

So the main features will be:
- Use python AI libraries to analyze data like pandas, numpy, scikit-learn, etc.
- Use Flask to create a web application that provides a user interface for the analytics
- Read two different data sources (CSV and SQLConnection) and then merge them or analyze them together
- Database connection will be only read-only without any write access.
- The Database connection will be done using SQLAlchemy and Flask-SQLAlchemy
- The application will be deployed as a docker container and docker compose will be used to run the application
- Target DB is Postgres



## Project Overview

This is a Flask-based web application named "junta-analytics". It's currently in a very early stage with a minimal structure:
- A basic Flask application in `app.py`
- Empty `static/` and `templates/` directories for future Flask assets

## Development Setup

### Prerequisites
- Python 3.x
- Flask

### Installation
To set up this project:

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Flask (since no requirements.txt exists yet)
pip install flask
```

### Running the Application
To run the application:

```bash
# From the project root
python app.py
```

The application will be available at http://127.0.0.1:5000/

## Project Structure

- `app.py` - The main Flask application entry point
- `static/` - Directory for static assets (CSS, JavaScript, images)
- `templates/` - Directory for HTML templates

## Best Practices

- Flask routes should be organized in a logical manner
- Template files should be placed in the `templates/` directory
- Static files (CSS, JS, images) should be placed in the `static/` directory
- Consider implementing a proper package structure as the application grows

## Core files and utility functions
