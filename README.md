# Agency Monitor

Early warning system for societal collapse based on Agency Calculus 4.3

## Overview
This system monitors "agency" (choice space) across 5 domains:
- Economic
- Political  
- Social
- Health
- Educational

## Current Status
- Measuring US brittleness at 9+/10
- Validating predictions with Haiti case study
- Historical validation: Chile 1973, Iran 1979, Soviet Union 1991, Rwanda 1994

## Quick Start
```bash
pip install -r requirements.txt
python api/main.py
```

## Architecture
- **ETL Layer**: Data collection from multiple sources
- **AI Layer**: XGBoost models for predictions
- **Agency Calculator**: Core brittleness calculations
- **API**: FastAPI backend
- **Dashboard**: Streamlit frontend

## Critical Timeline
3-month deadline for Haiti validation
