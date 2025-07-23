# api/database.py
import os
import uuid
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime,
    ForeignKey, Text, UniqueConstraint, Index, Numeric, CheckConstraint
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.dialects.postgresql import UUID, insert as pg_insert

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set!")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---

class Country(Base):
    __tablename__ = "countries"
    code = Column(String(3), primary_key=True)
    name = Column(String(100), nullable=False)
    region = Column(String(50))

class Indicator(Base):
    __tablename__ = "indicators"
    code = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    source = Column(String(100))
    # ... other fields

class Observation(Base):
    __tablename__ = "observations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.code"), nullable=False)
    indicator_code = Column(String(50), ForeignKey("indicators.code"), nullable=False)
    year = Column(Integer, nullable=False)
    value = Column(Numeric, nullable=False)
    dataset_version = Column(String(50))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('country_code', 'indicator_code', 'year', 'dataset_version', name='uq_observation'),
    )

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

def bulk_upsert(db: Session, table_name: str, data: List[Dict[str, Any]], index_elements: List[str]):
    """Generic bulk upsert function."""
    if not data:
        return {"affected_rows": 0}
    
    table = Base.metadata.tables[table_name]
    stmt = pg_insert(table).values(data)
    
    # Create the 'set_' dictionary for the ON CONFLICT clause
    update_cols = {
        col.name: getattr(stmt.excluded, col.name)
        for col in table.c
        if col.name not in index_elements and not col.primary_key
    }
    
    final_stmt = stmt.on_conflict_do_update(
        index_elements=index_elements,
        set_=update_cols
    )
    result = db.execute(final_stmt)
    return {"affected_rows": result.rowcount}


def bulk_upsert_observations(db: Session, observations: List[Dict[str, Any]]):
    """Specific bulk upsert for the observations table."""
    return bulk_upsert(db, 'observations', observations, ['country_code', 'indicator_code', 'year', 'dataset_version'])


# --- Initialization Script ---
def init_db():
    """Initialize database with tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created based on ORM models!")

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete.")