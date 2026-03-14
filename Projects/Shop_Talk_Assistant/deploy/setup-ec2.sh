#!/bin/bash
# ============================================================
# ShopTalk — EC2 Instance Setup Script
# ============================================================
# Run this ONCE on a fresh EC2 instance (Ubuntu 22.04 AMI)
#
# Recommended instance: g4dn.xlarge (T4 GPU, 16GB RAM)
# Also works on: t3.xlarge (CPU-only, slower LLM)
#
# Usage:
#   chmod +x deploy/setup-ec2.sh
#   ./deploy/setup-ec2.sh
# ============================================================

set -euo pipefail
echo "=========================================="
echo "ShopTalk — EC2 Setup"
echo "=========================================="

# --- 1. System updates ---
echo "[1/5] Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y curl git ca-certificates gnupg

# --- 2. Docker setup (official Docker repo — includes compose plugin) ---
echo "[2/5] Installing Docker from official repository..."

# Add Docker's official GPG key (remove stale key if present)
sudo install -m 0755 -d /etc/apt/keyrings
sudo rm -f /etc/apt/keyrings/docker.gpg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the Docker apt repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Verify docker compose is available
docker compose version

# --- 3. NVIDIA GPU drivers (skip on CPU-only instances) ---
if lspci | grep -qi nvidia; then
    echo "[3/5] Installing NVIDIA drivers + container toolkit..."
    
    # NVIDIA driver
    sudo apt-get install -y linux-headers-$(uname -r)
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo "  GPU detected and configured"
else
    echo "[3/5] No GPU detected — skipping NVIDIA setup (CPU-only mode)"
fi

# --- 4. Create project directory ---
echo "[4/5] Setting up project directory..."
PROJECT_DIR="$HOME/shoptalk"
mkdir -p "$PROJECT_DIR/data"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Upload your NB03/NB04 artifacts to $PROJECT_DIR/data/"
echo "     Required files:"
echo "       - rag_products.pkl (or products_with_prices.pkl)"
echo "       - rag_text_index.npy"
echo "       - rag_image_index.npy"
echo "       - rag_config.json"
echo ""
echo "     From your local machine:"
echo "       scp -i your-key.pem data/*.pkl data/*.npy data/*.json ubuntu@<EC2-IP>:~/shoptalk/data/"
echo ""
echo "  2. Clone the repo (or scp the project):"
echo "       cd ~/shoptalk"
echo "       git clone <your-repo-url> ."
echo ""
echo "  3. Deploy:"
echo "       cd ~/shoptalk"
echo "       ./deploy/deploy.sh"
echo ""
echo "  NOTE: Log out and back in for Docker group to take effect:"
echo "       exit"
echo "       ssh -i your-key.pem ubuntu@<EC2-IP>"
echo ""
