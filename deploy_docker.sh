#!/bin/bash

# Docker Deployment Script for Synthetic Medical Data GAN
# Built by Prashant Ambati

echo "🐳 Docker Deployment Script"
echo "Built by Prashant Ambati"
echo "=========================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is installed and running!"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t synthetic-medical-gan .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Failed to build Docker image!"
    exit 1
fi

# Run the container
echo "🚀 Starting the application..."
echo "Your app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"

docker run -p 8501:8501 synthetic-medical-gan