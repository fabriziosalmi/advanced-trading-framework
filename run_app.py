#!/usr/bin/env python3
"""
Advanced Trading Framework - Launch Script

Simple script to launch the Streamlit application with proper configuration.
"""

import subprocess
import sys
import os

# Add framework imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.config_validator import validate_configuration

def main():
    """Launch the trading framework application."""
    
    print("🚀 Launching Advanced Trading Framework...")
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Validate configuration first
    print("🔍 Validating configuration...")
    if not validate_configuration("config.yaml"):
        print("❌ Configuration validation failed. Please fix the issues above.")
        sys.exit(1)
    print("✅ Configuration validation passed")
    
    # Check if config exists
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml not found. Please ensure the configuration file exists.")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_path = os.path.join(script_dir, "trading_env", "bin", "activate")
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found. Please run setup_environment.py first.")
        sys.exit(1)
    
    # Launch Streamlit with virtual environment
    try:
        venv_python = os.path.join(script_dir, "trading_env", "bin", "python")
        cmd = [
            venv_python, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("📡 Starting Streamlit server...")
        print("🌐 App will be available at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Advanced Trading Framework...")
    except Exception as e:
        print(f"❌ Failed to launch application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()