from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    Date,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class WaterMeter(Base):
    """Water Meter model"""
    __tablename__ = 'water_meters'

    id = Column(Integer, primary_key=True)
    status = Column(Integer)
    activate_date = Column(DateTime)
    deactivate_date = Column(DateTime)
    number = Column(String(255))
    current_reading = Column(Integer)
    installation_date = Column(Date)
    expiration_date = Column(Date)
    neighborhood_id = Column(String(2))
    rate_id = Column(String(4))
    exclude_from_minga = Column(Boolean)


class Measure(Base):
    """Measure model"""
    __tablename__ = 'measures'

    id = Column(Integer, primary_key=True)
    previous_reading = Column(Integer)
    current_reading = Column(Integer)
    excess_consumption = Column(Integer)
    total_consumed = Column(Integer)
    finished_at = Column(DateTime)
    period_id = Column(Integer)
    water_meter_id = Column(Integer)
    code = Column(String(16))
    created_at = Column(DateTime)
    created_by_id = Column(Integer)
    operator_id = Column(Integer)
    status = Column(String(1))
    updated_at = Column(DateTime)
    updated_by_id = Column(Integer)


class Customer(Base):
    """Customer model"""
    __tablename__ = 'customers'

    id = Column(Integer, primary_key=True)
    status = Column(Integer)
    activate_date = Column(DateTime)
    deactivate_date = Column(DateTime)
    first_name = Column(String(255))
    last_name = Column(String(255))
    identification_type = Column(String(3))
    dni = Column(String(13))
    email = Column(String(254))
    phone = Column(String(255))
    address = Column(String(255))
    birth_date = Column(Date)
    is_older_adult = Column(Boolean)
    gender = Column(String(1))
    neighborhood_id = Column(String(2))
    full_name = Column(String(255))


class Neighborhood(Base):
    """Neighborhood model"""
    __tablename__ = 'neighborhoods'

    id = Column(String(2), primary_key=True)
    name = Column(String(255))


class Period(Base):
    """Period model"""
    __tablename__ = 'periods'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    start_date = Column(Date)
    end_date = Column(Date)
    collection_start_date = Column(Date)
    days_billed = Column(Integer)
    description = Column(Text)
    status = Column(String(12))


class Bill(Base):
    """Bill model"""
    __tablename__ = 'bills'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    status = Column(String(1))
    code = Column(String(16))
    start_date = Column(Date)
    end_date = Column(Date)
    expires_at = Column(DateTime)
    issued_at = Column(DateTime)
    days_billed = Column(Integer)
    billed = Column(Boolean)
    paid_at = Column(DateTime)
    canceled_at = Column(DateTime)
    created_by_id = Column(Integer)
    measure_id = Column(Integer)
    period_id = Column(Integer)
    updated_by_id = Column(Integer)
    water_meter_id = Column(Integer)
    marked_overdue_at = Column(DateTime)
    invoice_id = Column(Integer)
