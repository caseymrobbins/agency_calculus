# api/database.py
import os
import uuid
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, ForeignKey, Text, UniqueConstraint, Numeric
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.dialects.postgresql import UUID, insert as pg_insert
from sqlalchemy.schema import Table

# --- Configuration ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set!")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models (Corrected to match schema.sql) ---
class Country(Base):
    __tablename__ = "countries"
    country_code = Column(String(3), primary_key=True)
    country_name = Column(String(100), nullable=False, unique=True)
    region_code = Column(String(50), ForeignKey("regions.region_code"))
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

class Observation(Base):
    __tablename__ = "observations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False, index=True)
    indicator_code = Column(String(50), ForeignKey("indicators.indicator_code"), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    value = Column(Numeric)
    dataset_version = Column(String(50), nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('country_code', 'indicator_code', 'year', 'dataset_version', name='uq_observation'),
        {'postgresql_partition_by': 'RANGE (year)'}
    )

class AgencyScore(Base):
    __tablename__ = "agency_scores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    economic_agency = Column(Numeric)
    political_agency = Column(Numeric)
    social_agency = Column(Numeric)
    health_agency = Column(Numeric)
    educational_agency = Column(Numeric)
    calculation_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (UniqueConstraint('country_code', 'year', 'calculation_version', name='uq_agency_score'),)

class BrittlenessPrediction(Base):
    __tablename__ = "brittleness_predictions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False)
    target_year = Column(Integer, nullable=False, index=True)
    brittleness_score = Column(Numeric, nullable=False)
    confidence_lower = Column(Numeric)
    confidence_upper = Column(Numeric)
    risk_level = Column(String(20), nullable=False)
    model_version = Column(String(20), nullable=False)
    weighting_scheme = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (UniqueConstraint('country_code', 'target_year', 'model_version', 'weighting_scheme', name='uq_brittleness_prediction'),)

class IQANote(Base):
    __tablename__ = "iqa_notes"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    analyst = Column(String(100), nullable=False)
    note = Column(Text, nullable=False)
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def bulk_upsert(db: Session, table: Table, data: List], index_elements: List[str]):
    """Generic bulk upsert function."""
    if not data:
        return {"affected_rows": 0}
    
    stmt = pg_insert(table).values(data)
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
    db.commit()
    return {"affected_rows": result.rowcount}

def bulk_upsert_observations(db: Session, observations: List]):
    """Specific bulk upsert for the observations table."""
    return bulk_upsert(db, Observation.__table__, observations, ['country_code', 'indicator_code', 'year', 'dataset_version'])

def get_iqa_notes(country_code: str, year: Optional[int]) -> List]:
    with get_db() as db:
        query = db.query(IQANote).filter(IQANote.country_code == country_code)
        if year:
            query = query.filter(IQANote.year == year)
        notes = query.order_by(IQANote.created_at.desc()).all()
        return [{"analyst": n.analyst, "note": n.note, "year": n.year, "category": n.category, "created_at": n.created_at.isoformat()} for n in notes]

def save_iqa_note(note_data: Dict[str, Any]):
    with get_db() as db:
        new_note = IQANote(
            country_code=note_data['country_code'],
            year=note_data['year'],
            analyst=note_data['analyst'],
            note=note_data['note'],
            category=note_data.get('category', 'Other')
        )
        db.add(new_note)
        db.commit()