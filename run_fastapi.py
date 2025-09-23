#!/usr/bin/env python3
"""
Advanced Trading Framework - FastAPI Launch Script

Launch script for the FastAPI application with proper configuration.
"""

import subprocess
import sys
import os
import uvicorn

# Add framework imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.config_validator import validate_configuration

def main():
    """Launch the FastAPI trading framework application."""

    print("🚀 Launching Advanced Trading Framework (FastAPI)...")

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

    # Launch FastAPI with uvicorn
    try:
        print("📡 Starting FastAPI server...")
        print("🌐 App will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/api/docs")
        print("⏹️  Press Ctrl+C to stop the server")

        # Run uvicorn directly
        uvicorn.run(
            "fastapi_app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )

    except KeyboardInterrupt:
        print("\n👋 Shutting down Advanced Trading Framework...")
    except Exception as e:
        print(f"❌ Failed to launch application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()