# EC2 Deployment Guide: IMDB Spoiler Shield MLOps Stack

This guide provides instructions to deploy the entire IMDB Spoiler Shield MLOps infrastructure (Airflow, MLflow, Feast, FastAPI) on an AWS EC2 instance.

You can choose between the **Automated Approach** (Recommended) or the **Manual Approach**.

---

## 1. Instance Provisioning

*   **AMI**: Ubuntu Server 24.04 LTS
*   **Instance Type**: 
    *   Minimum: `t3.xlarge` (4 vCPU, 16GB RAM)
    *   Recommended (for BERT Training): `g4dn.xlarge` (GPU instance)
*   **Storage**: 50GB gp3 EBS volume.
*   **Security Group**: 
    *   Inbound: `22` (SSH), `8000` (FastAPI), `8080` (Airflow), `5000` (MLflow).

---

## 2. Option A: Automated Deployment (Recommended)

This method uses the provided helper scripts to configure the environment and launch the stack in minutes.

### Step 1: Clone Repository
Connect to your instance via SSH:
```bash
git clone <your-repository-url>
cd IMDB_Spoiler_Shield
```

### Step 2: System Setup
Run the setup script to update the system and install Docker.
```bash
chmod +x deployment/setup_ec2.sh
./deployment/setup_ec2.sh
```
**Important:** After this script finishes, **log out and log back in** to refresh your user permissions.

### Step 3: Launch Stack
Run the deployment script. It will prompt you for your AWS credentials (required for S3 artifact storage) and start the containers.
```bash
chmod +x deployment/deploy_stack.sh
./deployment/deploy_stack.sh
```

---

## 3. Option B: Manual Deployment

Follow these steps if you prefer to configure the server manually or need to debug the setup process.

### Step 1: Install Dependencies
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

### Step 2: Configure Environment
Create the `.env` file in the project root. This file injects secrets into the containers.
```bash
cat <<EOF > .env
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
S3_BUCKET=imdb-spoiler-shield-bucket
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
EOF
```

### Step 3: Launch Docker Compose
```bash
docker compose up -d --build
```

---

## 4. Verification Checklist

Regardless of the deployment method, verify the system health:

### Service Health
Run `docker compose ps` to ensure containers are running:
*   `postgres`: Healthy
*   `redis`: Healthy
*   `mlflow-server`: Up
*   `fastapi-app`: Up
*   `airflow-init`: Exited (success)
*   `airflow-webserver`: Up
*   `airflow-scheduler`: Up

### Manual Training Test
Verify that the Linux-based training pipeline can successfully run and upload artifacts to S3:
```bash
docker compose run --rm fastapi-app python src/train.py
```
Check your S3 bucket for `models/model.joblib`.

### API End-to-End Test
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{ 
           "movie_id": "tt0111161",
           "review_text": "The ending where he escapes through the tunnel is the best part!"
         }'
```

---

## 5. Remote UI Access (Tunneling)

To access the Airflow and MLflow UIs securely from your local Mac without exposing ports to the public internet, use SSH tunneling:

```bash
# From your local terminal
ssh -L 8080:localhost:8080 -L 5000:localhost:5000 -L 8000:localhost:8000 ubuntu@<EC2-Public-IP>
```

You can then visit:
*   **Airflow**: `http://localhost:8080` (airflow/airflow)
*   **MLflow**: `http://localhost:5000`
*   **FastAPI Docs**: `http://localhost:8000/docs`

