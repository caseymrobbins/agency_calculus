import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# --- ORM Models (should match your schema) ---
class Indicator(Base):
    __tablename__ = 'indicators'
    indicator_code = Column(String(100), primary_key=True)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    unit = Column(String(255))
    source = Column(String(255))
    topic = Column(String(255))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class Observation(Base):
    __tablename__ = 'observations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    country_code = Column(String(10), nullable=False)
    indicator_code = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    dataset_version = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # FIX: Changed the constraint name to match the SQL schema file
    __table_args__ = (UniqueConstraint('country_code', 'indicator_code', 'year', 'dataset_version', name='unique_observation'),)


# --- Database Connection ---
def get_db_session():
    """Get a new database session."""
    database_url = os.getenv('DATABASE_URL', 'postgresql://caseyrobbins@localhost:5432/agency_monitor')
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# --- Bulk Operations ---
def bulk_upsert_indicators(indicators: List[Dict[str, Any]]) -> int:
    """Bulk upsert indicators using ON CONFLICT DO UPDATE."""
    if not indicators: return 0
    
    session = get_db_session()
    try:
        stmt = pg_insert(Indicator).values(indicators)
        update_cols = {col.name: getattr(stmt.excluded, col.name) for col in Indicator.__table__.columns if not col.primary_key}
        final_stmt = stmt.on_conflict_do_update(index_elements=['indicator_code'], set_=update_cols)
        
        result = session.execute(final_stmt)
        session.commit()
        return result.rowcount
    except SQLAlchemyError as e:
        logger.error(f"Database error during indicator upsert: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def bulk_upsert_observations(observations: List[Dict[str, Any]]) -> int:
    """More robust bulk upsert for observations using SQLAlchemy's dialect-specific insert."""
    if not observations: return 0

    session = get_db_session()
    try:
        stmt = pg_insert(Observation).values(observations)
        
        update_cols = {
            'value': stmt.excluded.value,
            'dataset_version': stmt.excluded.dataset_version,
            'notes': stmt.excluded.notes,
            'updated_at': stmt.excluded.updated_at
        }
        
        # FIX: Changed the constraint name to match the SQL schema file
        final_stmt = stmt.on_conflict_do_update(
            constraint='unique_observation',
            set_=update_cols
        )
        
        result = session.execute(final_stmt)
        session.commit()
        return result.rowcount
    except SQLAlchemyError as e:
        logger.error(f"Database error during observation upsert: {e}")
        session.rollback()
        raise
    finally:
        session.close()