#!/bin/bash

# setup_gpu.sh
# Installs NVIDIA Container Toolkit on Ubuntu EC2 (g4dn.xlarge, etc.)
# Usage: sudo ./setup_gpu.sh

set -e

echo "🚀 Starting NVIDIA GPU Setup..."

# 1. Add NVIDIA Package Repositories
echo "📦 Adding NVIDIA repositories..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install Toolkit
echo "⬇️  Installing nvidia-container-toolkit..."
apt-get update
apt-get install -y nvidia-container-toolkit

# 3. Configure Docker
echo "⚙️  Configuring Docker runtime..."
nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker
echo "🔄 Restarting Docker..."
systemctl restart docker

echo "✅ GPU Setup Complete! You can now use 'driver: nvidia' in docker-compose."
