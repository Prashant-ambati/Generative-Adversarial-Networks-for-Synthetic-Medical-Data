#!/usr/bin/env python3
"""
Quick Demo Script for Synthetic Medical Data GAN
Built by Prashant Ambati
"""

import os
import sys
import subprocess
import argparse

def run_training_demo():
    """Run a quick training demonstration."""
    print("🚀 Starting GAN Training Demo...")
    print("Built by Prashant Ambati")
    print("-" * 50)
    
    # Run training with minimal epochs for demo
    cmd = [
        sys.executable, "src/train_gan.py",
        "--epochs", "10",
        "--batch_size", "32",
        "--eval_interval", "5",
        "--save_interval", "10"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training demo failed: {e}")
        return False
    
    return True

def run_dashboard():
    """Launch the Streamlit dashboard."""
    print("🌐 Launching Streamlit Dashboard...")
    print("Built by Prashant Ambati")
    print("-" * 50)
    
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard launch failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Synthetic Medical Data GAN Demo")
    parser.add_argument("--mode", choices=["install", "train", "dashboard", "all"], 
                       default="all", help="Demo mode to run")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 SYNTHETIC MEDICAL DATA GAN")
    print("Built by Prashant Ambati")
    print("=" * 60)
    
    if args.mode in ["install", "all"]:
        if not install_dependencies():
            return
    
    if args.mode in ["train", "all"]:
        if not run_training_demo():
            return
    
    if args.mode in ["dashboard", "all"]:
        run_dashboard()

if __name__ == "__main__":
    main()