# Use a compatible Python version
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files into the container
COPY . .

# Default command to run the app
CMD ["python", "app.py"]
