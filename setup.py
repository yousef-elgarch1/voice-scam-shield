#!/usr/bin/env python3
"""
🛡️ Voice Scam Interceptor - Setup Script
=========================================

This script sets up the Voice Scam Interceptor system with all dependencies
and configurations needed for real-time scam detection.

Usage:
    python setup.py install
    python setup.py develop  # For development mode
    python setup.py test     # Run tests
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_system_dependencies():
    """Install system-level dependencies"""
    print("📦 Installing system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y portaudio19-dev python3-pyaudio",
            "sudo apt-get install -y ffmpeg libsndfile1",
            "sudo apt-get install -y tkinter python3-tk"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd.split(), check=True)
                print(f"✅ {cmd}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: {cmd} failed: {e}")
    
    elif system == "darwin":  # macOS
        commands = [
            "brew install portaudio",
            "brew install ffmpeg"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd.split(), check=True)
                print(f"✅ {cmd}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: {cmd} failed (install Homebrew first): {e}")
    
    elif system == "windows":
        print("🪟 Windows detected - please install:")
        print("  1. Visual Studio C++ Build Tools")
        print("  2. FFmpeg (add to PATH)")
        print("  3. PortAudio (pip will handle this)")
    
    print("✅ System dependencies installation complete")

def install_python_packages():
    """Install Python packages"""
    print("🐍 Installing Python packages...")
    
    # Upgrade pip first
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install packages
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Python packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python packages: {e}")
        print("Trying individual package installation...")
        
        # Core packages that must be installed
        core_packages = [
            "torch",
            "torchaudio", 
            "transformers",
            "openai-whisper",
            "scikit-learn",
            "librosa",
            "pyaudio",
            "fastapi",
            "uvicorn",
            "matplotlib",
            "tkinter",
            "pyttsx3"
        ]
        
        for package in core_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")

def download_ai_models():
    """Download required AI models"""
    print("🤖 Downloading AI models...")
    
    try:
        import whisper
        print("📥 Downloading Whisper base model...")
        whisper.load_model("base")
        print("✅ Whisper model downloaded")
    except Exception as e:
        print(f"⚠️ Warning: Could not download Whisper model: {e}")
    
    try:
        from transformers import pipeline
        print("📥 Downloading sentiment analysis model...")
        pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        print("✅ Sentiment model downloaded")
    except Exception as e:
        print(f"⚠️ Warning: Could not download sentiment model: {e}")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "logs",
        "models", 
        "data",
        "recordings",
        "incidents",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")

def create_config_files():
    """Create default configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Main config
    config_content = """
# Voice Scam Interceptor Configuration
# ===================================

[audio]
sample_rate = 16000
chunk_size = 1024
channels = 1
record_seconds = 3

[detection]
risk_threshold_high = 70
risk_threshold_medium = 40
content_weight = 0.6
voice_weight = 0.3
sentiment_weight = 0.1

[response]
enable_confrontation = true
auto_block = true
log_incidents = true

[interface]
default_interface = gui
web_port = 8000
update_rate = 10

[models]
whisper_model = base
sentiment_model = cardiffnlp/twitter-roberta-base-sentiment-latest
"""
    
    with open("config/config.ini", "w") as f:
        f.write(config_content.strip())
    print("✅ Created config/config.ini")
    
    # Logging config
    logging_config = """
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "standard": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
  },
  "handlers": {
    "file": {
      "level": "INFO",
      "class": "logging.FileHandler",
      "filename": "logs/voice_interceptor.log",
      "formatter": "standard"
    },
    "console": {
      "level": "INFO", 
      "class": "logging.StreamHandler",
      "formatter": "standard"
    }
  },
  "loggers": {
    "": {
      "handlers": ["file", "console"],
      "level": "INFO",
      "propagate": false
    }
  }
}
"""
    
    with open("config/logging.json", "w") as f:
        f.write(logging_config.strip())
    print("✅ Created config/logging.json")

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    
    try:
        # Test audio system
        import pyaudio
        audio = pyaudio.PyAudio()
        info = audio.get_default_input_device_info()
        print(f"✅ Audio input device: {info['name']}")
        audio.terminate()
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
    
    try:
        # Test AI models
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
    
    try:
        # Test web framework
        from fastapi import FastAPI
        app = FastAPI()
        print("✅ FastAPI working")
    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")
    
    try:
        # Test GUI framework
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        print("✅ Tkinter working")
    except Exception as e:
        print(f"❌ Tkinter test failed: {e}")

def create_launcher_scripts():
    """Create launcher scripts for easy execution"""
    print("🚀 Creating launcher scripts...")
    
    # Windows batch file
    windows_launcher = """@echo off
echo 🛡️ Starting Voice Scam Interceptor...
cd /d "%~dp0"
python voice_interceptor.py
pause
"""
    
    with open("start_interceptor.bat", "w") as f:
        f.write(windows_launcher)
    print("✅ Created start_interceptor.bat")
    
    # Unix shell script
    unix_launcher = """#!/bin/bash
echo "🛡️ Starting Voice Scam Interceptor..."
cd "$(dirname "$0")"
python3 voice_interceptor.py
"""
    
    with open("start_interceptor.sh", "w") as f:
        f.write(unix_launcher)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start_interceptor.sh", 0o755)
    
    print("✅ Created start_interceptor.sh")

def main():
    """Main setup function"""
    print("""
🛡️ VOICE SCAM INTERCEPTOR SETUP
==============================
Advanced AI-Powered Real-Time Scam Protection System

This setup will install all dependencies and configure your system
for real-time voice scam detection and protection.
""")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            run_tests()
            return
        elif command == "clean":
            print("🧹 Cleaning installation...")
            # Add cleanup logic here
            return
    
    print("Starting installation...\n")
    
    # Installation steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing system dependencies", install_system_dependencies),
        ("Installing Python packages", install_python_packages),
        ("Creating directories", create_directories),
        ("Creating configuration files", create_config_files),
        ("Downloading AI models", download_ai_models),
        ("Creating launcher scripts", create_launcher_scripts),
        ("Running tests", run_tests)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            choice = input("Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                print("Setup aborted.")
                sys.exit(1)
    
    print(f"""
🎉 INSTALLATION COMPLETE!
========================

Your Voice Scam Interceptor is ready to protect you from scammer calls!

📋 Quick Start:
  1. Run: python voice_interceptor.py
  2. Or use: ./start_interceptor.sh (Unix) or start_interceptor.bat (Windows)
  3. Choose your preferred interface (GUI/Web/CLI)
  4. Click "START PROTECTION" to begin monitoring

🛡️ Features Ready:
  • Real-time voice analysis with AI
  • Advanced scam pattern detection  
  • Voice synthesis (deepfake) detection
  • Automatic scammer confrontation
  • Modern dashboard with live visualizations
  • Multi-interface support

⚠️ Important:
  • Grant microphone permissions when prompted
  • Ensure good internet connection for AI models
  • Keep the system running to intercept calls

🆘 Support:
  • Check logs/ directory for troubleshooting
  • Review config/config.ini for settings
  • Run 'python setup.py test' to verify installation

Stay safe from scammers! 🛡️
""")

if __name__ == "__main__":
    main()