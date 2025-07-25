# --- Base Stage --- 
# This stage installs all common dependencies for all services.
# It leverages Docker's build cache effectively.
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- API Stage ---
FROM base as api

# Copy application code
COPY api/ api/
COPY agency/ agency/
COPY config/ config/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Dashboard Stage ---
FROM base as dashboard

# Copy application code
COPY dashboard/ dashboard/
COPY api/ api/
COPY agency/ agency/
COPY config/ config/

# Set Python path
ENV PYTHONPATH=/app

# Streamlit configuration
RUN mkdir -p ~/.streamlit
RUN echo "[server]\nheadless = true\nport = 8501\nenableCORS = false\n" > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# --- ETL/Worker Stage ---
# This can be used for running ETL, training, and other offline tasks
FROM base as worker

# Copy relevant code
COPY etl/ etl/
COPY scripts/ scripts/
COPY ai/ ai/
COPY agency/ agency/
COPY config/ config/

# Set Python path
ENV PYTHONPATH=/app

# No default command; run via docker run -it worker python scripts/train_models.py