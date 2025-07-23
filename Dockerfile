# Dockerfile

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

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# --- API Stage ---
FROM python:3.11-slim as api

WORKDIR /app

# Copy installed packages from the base stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /root/.cache /root/.cache

# Copy application code
COPY api/./api/
COPY ai/./ai/
COPY agency/./agency/
COPY config/./config/
COPY models/./models/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" |

| exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# --- Dashboard Stage ---
FROM python:3.11-slim as dashboard

WORKDIR /app

# Copy installed packages from the base stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY dashboard.py.
COPY api/./api/
COPY agency/./agency/
COPY config/./config/

# Set Python path
ENV PYTHONPATH=/app

# Streamlit configuration
RUN mkdir -p ~/.streamlit
RUN echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\n' > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health |

| exit 1

# Default command
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]


# --- ETL/Worker Stage ---
# This can be used for running ETL, training, and other offline tasks
FROM python:3.11-slim as worker

WORKDIR /app

# Copy installed packages from the base stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /root/.cache /root/.cache

# Copy all code needed for tasks
COPY..

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden by docker-compose)
CMD ["python", "-m", "etl.populate_metadata"]