# Junta Analytics - Water Quality Monitoring System

A Flask-based web application for water quality monitoring and consumption analytics. This system processes water meter readings and rainfall data to detect anomalies, generate forecasts, and visualize consumption patterns.

## Features

- Data integration from PostgreSQL database (meter readings) and CSV files (rainfall)
- Data processing and fusion of consumption and precipitation data
- Anomaly detection using Isolation Forest algorithm
- Consumption forecasting
- Interactive dashboards and visualization
- API endpoints for data access
- Consumption validation form

## Tech Stack

- **Backend**: Flask, SQLAlchemy
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Frontend**: Tailwind CSS
- **Database**: PostgreSQL
- **Deployment**: Docker, Docker Compose

## Project Structure

```
junta-analytics/
├── app.py                  # Main application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose services
├── init.sql                # Database initialization
├── data/                   # Data directory
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── external/           # External data files
├── models/                 # Machine learning models
├── static/                 # Static assets
├── templates/              # HTML templates
├── utils/                  # Utility functions
├── services/               # Service layer
└── controllers/            # Controllers/routes
```

## Setup and Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/junta-analytics.git
   cd junta-analytics
   ```

2. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```

3. Update the `.env` file with your configuration.

4. Start the application with Docker Compose:
   ```bash
   docker-compose up -d
   ```

5. Access the application at http://localhost:5000

### Manual Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/junta-analytics.git
   cd junta-analytics
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```

5. Update the `.env` file with your configuration.

6. Start the application:
   ```bash
   flask run
   ```

7. Access the application at http://localhost:5000

## API Endpoints

- `/api/meter-readings` - Get all meter readings
- `/api/consumption-rainfall` - Get merged consumption and rainfall data
- `/api/forecast` - Get consumption forecasts
- `/api/check-consumption` - Check if a consumption value is anomalous

## Data Sources

- Meter readings: PostgreSQL database
- Rainfall data: CSV file (dekadal rainfall indicators)

## License

[MIT License](LICENSE)