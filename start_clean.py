"""
Startup script for Industrial Safety QA Service
Initializes system components with clean data loading
"""

import sys
import os
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.app import app, init_components
from src.utils.config import Config

def main():
    print("Starting QA Service...")
    
    # Initialize components
    print("Initializing system components...")
    init_components()
    
    print("Initialization complete")
    
    # Show startup information
    print(f"Server Configuration:")
    print(f"   Host: {Config.FLASK_HOST}")
    print(f"   Port: {Config.FLASK_PORT}")
    print(f"   Debug: {Config.FLASK_DEBUG}")
    
    print(f"\nAPI Endpoints:")
    print(f"   Home: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/")
    print(f"   Ask: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/ask (POST)")
    print(f"   Status: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/status (GET)")
    
    print(f"\nWeb interface: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )

if __name__ == "__main__":
    main()