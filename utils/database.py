import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# Initialize SQLAlchemy
db = SQLAlchemy()

def get_meter_readings():
    """
    Fetch meter readings from the database
    """
    query = """
    SELECT 
        meter_id, 
        reading_date, 
        consumption, 
        latitude, 
        longitude, 
        location_name
    FROM meter_readings
    ORDER BY reading_date DESC
    """
    
    try:
        result = db.session.execute(text(query))
        readings_data = [dict(row._mapping) for row in result]
        return pd.DataFrame(readings_data)
    except Exception as e:
        print(f"Error fetching meter readings: {e}")
        return pd.DataFrame()

def get_readings_by_period(start_date, end_date):
    """
    Fetch meter readings for a specific time period
    """
    query = """
    SELECT 
        meter_id, 
        reading_date, 
        consumption, 
        latitude, 
        longitude, 
        location_name
    FROM meter_readings
    WHERE reading_date BETWEEN :start_date AND :end_date
    ORDER BY reading_date
    """
    
    try:
        result = db.session.execute(
            text(query), 
            {"start_date": start_date, "end_date": end_date}
        )
        readings_data = [dict(row._mapping) for row in result]
        return pd.DataFrame(readings_data)
    except Exception as e:
        print(f"Error fetching meter readings by period: {e}")
        return pd.DataFrame()

def get_aggregate_consumption_by_dekad():
    """
    Aggregate consumption data by dekad periods (10-day periods)
    """
    query = """
    SELECT 
        CASE
            WHEN EXTRACT(DAY FROM reading_date) <= 10 THEN 
                DATE_TRUNC('month', reading_date) + INTERVAL '0 days'
            WHEN EXTRACT(DAY FROM reading_date) <= 20 THEN 
                DATE_TRUNC('month', reading_date) + INTERVAL '10 days'
            ELSE 
                DATE_TRUNC('month', reading_date) + INTERVAL '20 days'
        END AS dekad_period,
        SUM(consumption) as total_consumption,
        AVG(consumption) as avg_consumption,
        COUNT(*) as reading_count
    FROM meter_readings
    GROUP BY dekad_period
    ORDER BY dekad_period
    """
    
    try:
        result = db.session.execute(text(query))
        dekad_data = [dict(row._mapping) for row in result]
        return pd.DataFrame(dekad_data)
    except Exception as e:
        print(f"Error aggregating consumption by dekad: {e}")
        return pd.DataFrame()