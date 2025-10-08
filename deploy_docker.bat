@echo off
REM Docker Deployment Script for Synthetic Medical Data GAN
REM Built by Prashant Ambati

echo 🐳 Docker Deployment Script
echo Built by Prashant Ambati
echo ==========================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is installed and running!

REM Build the Docker image
echo 🔨 Building Docker image...
docker build -t synthetic-medical-gan .

if %errorlevel% neq 0 (
    echo ❌ Failed to build Docker image!
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!

REM Run the container
echo 🚀 Starting the application...
echo Your app will be available at: http://localhost:8501
echo Press Ctrl+C to stop the application

docker run -p 8501:8501 synthetic-medical-gan