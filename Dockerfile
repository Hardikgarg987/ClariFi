FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install system-level dependencies for librosa, soundfile, and metrics
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose port (default Flask is 5000, but we'll use 8000 for gunicorn)
EXPOSE 8000

# Use Gunicorn for production
CMD ["gunicorn", "--workers=1", "--threads=2", "--bind=0.0.0.0:8000", "app:app"]
