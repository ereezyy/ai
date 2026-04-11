#!/bin/bash

# 💥 THE MACHINE GOD AWAKENS: LINODE INITIATION 💥
# This script forcibly bends the server to the will of the AI Toolkit.

echo "⚡ INITIATING LINODE DOMINATION PROTOCOL ⚡"
echo "🩸 PREPARING THE VESSEL..."

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install base dependencies
echo "🔥 FORGING BASE DEPENDENCIES..."
sudo apt-get install -y python3 python3-pip python3-venv git htop screen

# Create the environment
echo "🌀 SUMMONING THE VIRTUAL ENVIRONMENT..."
python3 -m venv ai_env
source ai_env/bin/activate

# Install the Toolkit
echo "⚙️ BINDING THE TOOLKIT TO THE ENVIRONMENT..."
pip install --upgrade pip
pip install -e .

# Final Awakening
echo "=================================================="
echo "💥 LINODE DOMINATION PROTOCOL COMPLETE 💥"
echo "=================================================="
echo "The vessel is ready. To summon the Machine God:"
echo "1. source ai_env/bin/activate"
echo "2. export GROQ_API_KEY='your-key-here'"
echo "3. ai-toolkit awaken"
echo "=================================================="
