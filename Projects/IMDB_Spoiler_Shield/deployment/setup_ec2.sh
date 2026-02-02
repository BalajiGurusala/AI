#!/bin/bash

# setup_ec2.sh
# Automates the initial setup of an Ubuntu EC2 instance for the IMDB Spoiler Shield stack.
# Usage: ./setup_ec2.sh

set -e # Exit on error

echo "🚀 Starting EC2 Initialization for IMDB Spoiler Shield..."

# 1. System Updates
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git python3-pip ca-certificates curl gnupg

# 2. Install Docker
echo "🐳 Installing Docker & Docker Compose..."
if ! command -v docker &> /dev/null;
then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
    echo "✅ Docker is already installed."
fi

# 3. Permissions
echo "🔑 Configuring user permissions..."
if ! groups $USER | grep -q "\bdocker\b";
then
    sudo usermod -aG docker $USER
    echo "⚠️  User added to docker group. You must LOG OUT and LOG BACK IN for this to take effect."
else
    echo "✅ User already in docker group."
fi

echo "✨ EC2 Setup Complete! Please reconnect to your session."
