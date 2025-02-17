#!/bin/bash
source venv/bin/activate  # Activate virtual environment (if using one)
exec gunicorn -w 4 -b 0.0.0.0:8080 app:app  # Start Flask using Gunicorn
