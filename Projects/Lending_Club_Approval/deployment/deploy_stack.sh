#!/bin/bash

# deploy_stack.sh
# Automates the configuration and launch of the MLOps stack.

set -e

# Default values
DEFAULT_BUCKET_NAME="IK-lending-club-bucket"

echo "🚀 Starting MLOps Stack Deployment..."

# Ensure we are in the project root
if [ ! -f "docker-compose.yaml" ]; then
    echo "❌ Error: docker-compose.yaml not found. Please run this script from the project root."
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

    cat <<EOF > .env
AWS_ACCESS_KEY_ID=${AWS_KEY}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET}
DEFAULT_BUCKET=${BUCKET}
UPLOAD_TO_S3=true
PROJECT_HOME=/opt/airflow
EOF
    echo "✅ .env file created."
else
    echo "✅ .env file found. Skipping configuration."
fi

# 2. Launch Stack
echo "🐳 Building and starting containers..."
docker compose up -d --build

# 3. Health Check
echo "⏳ Waiting for services to stabilize..."
sleep 10
docker compose ps

echo ""
echo "🎉 Stack is Live!"
echo "Note: If running on EC2, verify Security Groups are open."
echo "➡️  Airflow: http://<PUBLIC-IP>:8080 (admin/admin)"
echo "➡️  MLflow:  http://<PUBLIC-IP>:5000"
echo "➡️  FastAPI: http://<PUBLIC-IP>:8000/docs"
