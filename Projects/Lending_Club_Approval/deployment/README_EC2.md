# EC2 Deployment Guide: Lending Club MLOps Stack

This guide provides step-by-step instructions to deploy the entire Lending Club MLOps infrastructure (Airflow, MLflow, Feast, FastAPI) on an AWS EC2 instance.

---

## 1. Instance Provisioning

*   **AMI**: Ubuntu Server 24.04 LTS
*   **Instance Type**: 
    *   Minimum: `t3.xlarge` (4 vCPU, 16GB RAM)
    *   Recommended (for GPU/Distributed Tuning): `g4dn.xlarge`
*   **Storage**: 50GB gp3 EBS volume.
*   **Security Group**: 
    *   Inbound: `22` (SSH), `8000` (FastAPI), `8080` (Airflow), `5000` (MLflow).

---

## 2. Environment Setup

Connect to your instance via SSH and run the following setup script:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Grant Docker permissions to the user
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3. Clone and Configure

```bash
# Clone the repository
git clone <your-repository-url>
cd Lending_Club_Approval

# Create the .env file for sensitive configurations
cat <<EOF > .env
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
DEFAULT_BUCKET=IK-lending-club-bucket
UPLOAD_TO_S3=true
PROJECT_HOME=/opt/airflow
EOF
```

---

## 4. Launch Infrastructure

Deploy all services using Docker Compose:

```bash
docker compose up -d --build
```

---

## 5. Verification Checklist

### 5.1 Service Health
Run `docker compose ps` to ensure all 6 containers are running:
*   `postgres`: Healthy
*   `mlflow`: Up
*   `fastapi-app`: Up
*   `airflow-init`: Exited (success)
*   `airflow-webserver`: Up
*   `airflow-scheduler`: Up

### 5.2 Manual Training Test
Verify that the Linux-based training pipeline can successfully run and upload artifacts to S3:
```bash
docker compose run --rm fastapi-app python src/train_ray.py
```
Check your S3 bucket for `artifacts/best_model.pkl`.

### 5.3 API End-to-End Test
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "borrower_id": 1071795,
           "loan_amnt": 5600.0,
           "term": "60 months",
           "purpose": "small_business"
         }'
```

---

## 6. Remote UI Access (Tunneling)

To access the Airflow and MLflow UIs securely from your local Mac without exposing ports to the public internet, use SSH tunneling:

```bash
# From your local terminal
ssh -L 8080:localhost:8080 -L 5000:localhost:5000 -L 8000:localhost:8000 ubuntu@<EC2-Public-IP>
```

You can then visit:
*   **Airflow**: `http://localhost:8080` (admin/admin)
*   **MLflow**: `http://localhost:5000`
*   **FastAPI Docs**: `http://localhost:8000/docs`

```
