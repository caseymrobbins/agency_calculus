"""
Database models and utilities for the Agency Monitor project.

Note: For schema migrations, integrate Alembic:
- Install: pip install alembic
- Init: alembic init migrations
- Autogenerate: alembic revision --autogenerate -m "Initial schema"
- Apply: alembic upgrade head
This ensures DevOps-friendly deployments with CI/CD (e.g., GitHub Actions running migrations).
"""

import os
import uuid
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, ForeignKey, Text, UniqueConstraint, Numeric,
    Enum as SQLEnum, JSON, event, Index
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID, insert as pg_insert, JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import Table as SATable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL, echo=False)  # Set echo=True for debugging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums for categorical fields
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

# --- ORM Models ---

class Region(Base):
    __tablename__ = "regions"
    region_code = Column(String(50), primary_key=True)
    region_name = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    countries = relationship("Country", back_populates="region")

    def __repr__(self):
        return f"<Region(code={self.region_code}, name={self.region_name})>"

class Country(Base):
    __tablename__ = "countries"
    country_code = Column(String(3), primary_key=True)  # e.g., 'USA'
    country_name = Column(String(100), nullable=False, unique=True)
    region_code = Column(String(50), ForeignKey("regions.region_code"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    region = relationship("Region", back_populates="countries")
    observations = relationship("Observation", back_populates="country")
    agency_scores = relationship("AgencyScore", back_populates="country")
    brittleness_predictions = relationship("BrittlenessPrediction", back_populates="country")
    iqa_notes = relationship("IQANote", back_populates="country")

    def __repr__(self):
        return f"<Country(code={self.country_code}, name={self.country_name})>"

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

    # Relationship
    observations = relationship("Observation", back_populates="indicator")

    def __repr__(self):
        return f"<Indicator(code={self.indicator_code}, name={self.indicator_name})>"

class Observation(Base):
    __tablename__ = "observations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False)
    indicator_code = Column(String(50), ForeignKey("indicators.indicator_code"), nullable=False)
    year = Column(Integer, nullable=False)
    value = Column(Numeric)
    dataset_version = Column(String(50), nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    country = relationship("Country", back_populates="observations")
    indicator = relationship("Indicator", back_populates="observations")

    __table_args__ = (
        UniqueConstraint('country_code', 'indicator_code', 'year', 'dataset_version', name='uq_observation'),
        Index('ix_observation_country_year', 'country_code', 'year'),  # Composite index for queries
    )

    def __repr__(self):
        return f"<Observation(id={self.id}, country={self.country_code}, indicator={self.indicator_code}, year={self.year})>"

class AgencyScore(Base):
    __tablename__ = "agency_scores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False)
    year = Column(Integer, nullable=False)
    economic_agency = Column(Numeric)
    political_agency = Column(Numeric)
    social_agency = Column(Numeric)
    health_agency = Column(Numeric)
    educational_agency = Column(Numeric)
    calculation_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    country = relationship("Country", back_populates="agency_scores")

    __table_args__ = (
        UniqueConstraint('country_code', 'year', 'calculation_version', name='uq_agency_score'),
        Index('ix_agency_score_country_year', 'country_code', 'year'),
    )

    def __repr__(self):
        return f"<AgencyScore(id={self.id}, country={self.country_code}, year={self.year})>"

class BrittlenessPrediction(Base):
    __tablename__ = "brittleness_predictions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False)
    target_year = Column(Integer, nullable=False)
    brittleness_score = Column(Numeric, nullable=False)
    confidence_interval_low = Column(Numeric)
    confidence_interval_high = Column(Numeric)
    risk_level = Column(SQLEnum(RiskLevel), nullable=True)
    trajectory = Column(String(50))
    days_to_critical = Column(Integer)
    top_risk_factors = Column(JSONB)  # Use JSONB for PostgreSQL
    model_version = Column(String(50), nullable=False)
    weighting_scheme = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    country = relationship("Country", back_populates="brittleness_predictions")

    __table_args__ = (
        UniqueConstraint('country_code', 'target_year', 'model_version', 'weighting_scheme', name='uq_brittleness_prediction'),
        Index('ix_brittleness_prediction_country_year', 'country_code', 'target_year'),
    )

    def __repr__(self):
        return f"<BrittlenessPrediction(id={self.id}, country={self.country_code}, year={self.target_year})>"

class IQANote(Base):
    __tablename__ = "iqa_notes"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), ForeignKey("countries.country_code"), nullable=False)
    year = Column(Integer, nullable=False)
    analyst = Column(String(100), nullable=False)
    note = Column(Text, nullable=False)
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    country = relationship("Country", back_populates="iqa_notes")

    __table_args__ = (
        Index('ix_iqa_note_country_year', 'country_code', 'year'),
    )

    def __repr__(self):
        return f"<IQANote(id={self.id}, country={self.country_code}, year={self.year})>"

class IngestionLog(Base):
    __tablename__ = "ingestion_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)
    records_processed = Column(Integer)
    records_inserted = Column(Integer)
    records_updated = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<IngestionLog(id={self.id}, source={self.source}, status={self.status})>"

# --- Database Functions ---

def bulk_upsert(db: Session, table: SATable, data: List[Dict[str, Any]], index_elements: List[str], update_cols: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Performs bulk upsert on the given table.
    Returns dict with 'inserted' and 'updated' counts (approximate for PostgreSQL).
    """
    if not data:
        return {'inserted': 0, 'updated': 0}
    
    stmt = pg_insert(table).values(data)
    if update_cols:
        update_dict = {c: stmt.excluded[c] for c in update_cols}
        final_stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=update_dict
        )
    else:
        final_stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
    
    try:
        result = db.execute(final_stmt)
        affected = result.rowcount
        # Approximate split: PostgreSQL doesn't distinguish, but assume all updates if conflict
        inserted = affected if not update_cols else 0  # Refine with query if needed
        updated = affected if update_cols else 0
        db.commit()
        return {'inserted': inserted, 'updated': updated}
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Bulk upsert failed: {str(e)}")
        raise

def bulk_upsert_observations(db: Session, observations: List[Dict[str, Any]]) -> Dict[str, int]:
    """Specific bulk upsert for the observations table."""
    return bulk_upsert(db, Observation.__table__, observations, ['country_code', 'indicator_code', 'year', 'dataset_version'])

def get_iqa_notes(country_code: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieves IQA notes for a country and optional year.
    Validates country_code to uppercase.
    """
    country_code = country_code.upper()  # Standardize
    with get_db() as db:
        query = db.query(IQANote).filter(IQANote.country_code == country_code)
        if year:
            query = query.filter(IQANote.year == year)
        notes = query.order_by(IQANote.created_at.desc()).all()
        return [
            {'analyst': n.analyst, 'note': n.note, 'year': n.year, 'category': n.category, 'created_at': n.created_at.isoformat()}
            for n in notes
        ]

def save_iqa_note(note_data: Dict[str, Any]):
    """
    Saves a new IQA note with validation.
    """
    required_fields = ['country_code', 'year', 'analyst', 'note']
    for field in required_fields:
        if field not in note_data or not note_data[field]:
            raise ValueError(f"Missing or empty required field: {field}")
    
    with get_db() as db:
        new_note = IQANote(
            country_code=note_data['country_code'].upper(),
            year=note_data['year'],
            analyst=note_data['analyst'],
            note=note_data['note'],
            category=note_data.get('category', 'Other')
        )
        db.add(new_note)
        db.commit()
        logger.info(f"Saved IQA note for {new_note.country_code} - {new_note.year}")