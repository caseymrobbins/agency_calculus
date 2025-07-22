"""
Database models and connection management for Agency Monitor
Using SQLAlchemy ORM for PostgreSQL interaction
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import uuid

from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime,
    ForeignKey, Text, Date, UniqueConstraint,
    Index, CheckConstraint, Numeric
)
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID, insert as pg_insert

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/agency_monitor")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models (Aligned with create_schema.sql) ---

class Country(Base):
    __tablename__ = "countries"
    country_code = Column(String(3), primary_key=True)
    country_name = Column(String(100), nullable=False)
    region = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Indicator(Base):
    __tablename__ = "indicators"
    indicator_code = Column(String(50), primary_key=True)
    indicator_name = Column(String(255), nullable=False)
    description = Column(Text)
    source = Column(String(100), nullable=False)
    access_method = Column(String(10), nullable=False)
    domain = Column(String(20))
    unit_of_measure = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        CheckConstraint(access_method.in_(['API', 'Bulk'])),
        CheckConstraint(domain.in_(['economic', 'political', 'social', 'health', 'educational', 'composite'])),
    )

class Observation(Base):
    __tablename__ = "observations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False)
    indicator_code = Column(String(50), ForeignKey("indicators.indicator_code"), nullable=False)
    year = Column(Integer, nullable=False)
    value = Column(Numeric)
    dataset_version = Column(String(50), nullable=False)
    data_quality = Column(Numeric)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('country_code', 'indicator_code', 'year', 'dataset_version', name='unique_observation'),
        Index('idx_observations_country_year', 'country_code', 'year'),
    )

class AgencyScore(Base):
    __tablename__ = "agency_scores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False)
    year = Column(Integer, nullable=False)
    indicator_code = Column(String(50), ForeignKey("indicators.indicator_code"), nullable=False)
    score = Column(Numeric)
    calculation_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('country_code', 'indicator_code', 'year', 'calculation_version', name='unique_agency_score'),
    )

# ... other models like BrittlenessPrediction, IQANote, etc. would follow the same pattern ...

# --- Database Session Management ---

@contextmanager
def get_db() -> Session:
    """Provide a transactional scope for database operations."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# --- High-Performance Database Functions ---

def bulk_upsert_observations(db: Session, observations: List[Dict[str, Any]]):
    """
    Performs a high-performance bulk insert/update for observations using
    PostgreSQL's ON CONFLICT DO UPDATE feature.
    """
    if not observations:
        return {"inserted": 0, "updated": 0}

    stmt = pg_insert(Observation).values(observations)
    
    update_dict = {
        'value': stmt.excluded.value,
        'data_quality': stmt.excluded.data_quality,
        'notes': stmt.excluded.notes,
        'updated_at': datetime.utcnow(),
    }
    
    final_stmt = stmt.on_conflict_do_update(
        index_elements=['country_code', 'indicator_code', 'year', 'dataset_version'],
        set_=update_dict
    )
    
    result = db.execute(final_stmt)
    db.commit() # Commit the transaction
    # Note: result.rowcount doesn't distinguish between inserts and updates, it's the total affected rows.
    return {"affected_rows": result.rowcount}

# --- Initialization Script ---
def init_db():
    """Initialize database with tables. Use with caution."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created based on ORM models!")

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete.")