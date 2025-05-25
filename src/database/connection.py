import logging
from typing import Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseManager:
    """Database connection and query manager"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.engine = None
        self.Session = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(
                self.config.DATABASE_URI,
                echo=False,  # Set to True for SQL logging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_session(self):
        """Get a new database session"""
        return self.Session()

    def execute_query(self, query: str,
                      params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params=params)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def get_water_meters_data(self) -> pd.DataFrame:
        """Get all water meters with customer information"""
        query = """
        SELECT 
            wm.id as water_meter_id,
            wm.number as meter_number,
            wm.current_reading,
            wm.installation_date,
            wm.neighborhood_id,
            wm.rate_id,
            wm.status as meter_status,
            n.name as neighborhood_name,
            c.id as customer_id,
            c.first_name,
            c.last_name,
            c.full_name,
            c.dni,
            c.phone,
            c.address,
            wmm.relation_type
        FROM water_meters wm
        LEFT JOIN neighborhoods n ON wm.neighborhood_id = n.id
        LEFT JOIN water_meter_members wmm ON wm.id = wmm.water_meter_id
        LEFT JOIN customers c ON wmm.customer_id = c.id
        WHERE wm.status = 1 AND wmm.status = 1
        ORDER BY wm.id
        """
        return self.execute_query(query)

    def get_measures_data(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Get measures data with optional date filtering"""
        base_query = """
        SELECT 
            m.id as measure_id,
            m.water_meter_id,
            m.period_id,
            m.previous_reading,
            m.current_reading,
            m.total_consumed,
            m.excess_consumption,
            m.created_at,
            m.finished_at,
            m.status as measure_status,
            m.code as measure_code,
            p.name as period_name,
            p.start_date as period_start,
            p.end_date as period_end,
            p.days_billed,
            wm.number as meter_number,
            wm.neighborhood_id,
            n.name as neighborhood_name
        FROM measures m
        JOIN periods p ON m.period_id = p.id
        JOIN water_meters wm ON m.water_meter_id = wm.id
        LEFT JOIN neighborhoods n ON wm.neighborhood_id = n.id
        WHERE m.status = 'F'  -- Only finished measures
        """

        params = {}
        if start_date:
            base_query += " AND p.start_date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            base_query += " AND p.end_date <= :end_date"
            params['end_date'] = end_date

        base_query += " ORDER BY m.created_at DESC"

        return self.execute_query(base_query, params)

    def get_bills_data(self) -> pd.DataFrame:
        """Get bills data with customer and period information"""
        query = """
        SELECT 
            b.id as bill_id,
            b.water_meter_id,
            b.measure_id,
            b.period_id,
            b.status as bill_status,
            b.code as bill_code,
            b.start_date,
            b.end_date,
            b.days_billed,
            b.created_at,
            b.paid_at,
            bd.total_price,
            p.name as period_name,
            wm.number as meter_number,
            wm.neighborhood_id,
            n.name as neighborhood_name
        FROM bills b
        LEFT JOIN bill_details bd ON b.id = bd.bill_id
        JOIN periods p ON b.period_id = p.id
        JOIN water_meters wm ON b.water_meter_id = wm.id
        LEFT JOIN neighborhoods n ON wm.neighborhood_id = n.id
        ORDER BY b.created_at DESC
        """
        return self.execute_query(query)

    def get_neighborhoods_stats(self) -> pd.DataFrame:
        """Get neighborhood statistics"""
        query = """
        SELECT 
            n.id as neighborhood_id,
            n.name as neighborhood_name,
            COUNT(DISTINCT wm.id) as total_meters,
            COUNT(DISTINCT c.id) as total_customers,
            AVG(m.total_consumed) as avg_consumption,
            SUM(m.total_consumed) as total_consumption,
            COUNT(m.id) as total_measures
        FROM neighborhoods n
        LEFT JOIN water_meters wm ON n.id = wm.neighborhood_id
        LEFT JOIN water_meter_members wmm ON wm.id = wmm.water_meter_id
        LEFT JOIN customers c ON wmm.customer_id = c.id
        LEFT JOIN measures m ON wm.id = m.water_meter_id
        WHERE wm.status = 1 AND wmm.status = 1 AND m.status = 'F'
        GROUP BY n.id, n.name
        ORDER BY total_consumption DESC
        """
        return self.execute_query(query)

    def get_consumption_by_period(self) -> pd.DataFrame:
        """Get consumption aggregated by period"""
        query = """
        SELECT 
            p.id as period_id,
            p.name as period_name,
            p.start_date,
            p.end_date,
            p.days_billed,
            COUNT(m.id) as total_measures,
            SUM(m.total_consumed) as total_consumption,
            AVG(m.total_consumed) as avg_consumption,
            MIN(m.total_consumed) as min_consumption,
            MAX(m.total_consumed) as max_consumption,
            STDDEV(m.total_consumed) as std_consumption
        FROM periods p
        LEFT JOIN measures m ON p.id = m.period_id
        WHERE m.status = 'F'
        GROUP BY p.id, p.name, p.start_date, p.end_date, p.days_billed
        ORDER BY p.start_date DESC
        """
        return self.execute_query(query)

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
