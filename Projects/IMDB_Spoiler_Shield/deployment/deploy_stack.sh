#!/bin/bash

# deploy_stack.sh
# Automates the configuration and launch of the IMDB Spoiler Shield MLOps stack.

set -e

# Default values
DEFAULT_BUCKET_NAME="imdb-spoiler-shield-bucket"

echo "🚀 Starting IMDB Spoiler Shield Stack Deployment..."

# Ensure we are in the project root
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

# 1. Environment Configuration
if [ ! -f .env ]; then
    echo "⚙️  .env file not found. Let's configure it."
    
    read -p "Enter AWS Access Key ID: " AWS_KEY
    read -s -p "Enter AWS Secret Access Key: " AWS_SECRET
    echo ""
    read -p "Enter S3 Bucket Name [${DEFAULT_BUCKET_NAME}]: " BUCKET
    BUCKET=${BUCKET:-$DEFAULT_BUCKET_NAME}

    # IMDB project uses S3_BUCKET env var
    cat <<EOF > .env
AWS_ACCESS_KEY_ID=${AWS_KEY}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET}
S3_BUCKET=${BUCKET}
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
EOF
    echo "✅ .env file created."
else
    echo "✅ .env file found. Skipping configuration."
fi

# 2. Permissions and Directories
echo "📂 Preparing directories and permissions..."
# Create necessary directories mapped in docker-compose
mkdir -p dags src data logs plugins feature_repo models
# Ensure the Docker 'airflow' user (UID 50000) can write to these
sudo chown -R 50000:0 logs dags plugins feature_repo data models
sudo chmod -R 775 logs dags plugins feature_repo data models

# 3. Launch Stack
echo "🐳 Building and starting containers..."
docker compose up -d --build

# 4. Health Check
echo "⏳ Waiting for services to stabilize..."
sleep 15
docker compose ps

echo ""
echo "🎉 Stack is Live!"
echo "Note: If running on EC2, verify Security Groups are open."
echo "➡️  Airflow: http://<PUBLIC-IP>:8080 (airflow/airflow)"
echo "➡️  MLflow:  http://<PUBLIC-IP>:5000"
echo "➡️  FastAPI: http://<PUBLIC-IP>:8000/docs"
