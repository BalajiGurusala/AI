FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g. for Feast/GCC)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set PYTHONPATH so src modules are discoverable
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
