"""
Streamlit App Entry Point for Deployment
Built by Prashant Ambati
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()