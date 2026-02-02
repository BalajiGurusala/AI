FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any (none strictly needed for basic sklearn/fastapi but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# We don't copy ../src here because we will mount it in docker-compose
# This ensures code changes in src/ are reflected immediately if needed
# However, for a production build, you would COPY ../src /app/src

# Download NLTK data required for preprocessing (stopwords)
RUN python -m nltk.downloader stopwords

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
