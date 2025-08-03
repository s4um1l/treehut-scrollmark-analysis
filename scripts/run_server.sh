#!/bin/bash

# Instagram Scrollmark Analysis - Backend Server Runner
# This script starts the FastAPI backend with proper environment setup

set -e  # Exit on any error

echo "🚀 Instagram Scrollmark Analysis - Backend Server"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "backend/pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if database exists
if [ ! -f "reports/enriched_instagram_data.sqlite" ]; then
    echo "❌ Database not found. Please run the pipeline first:"
    echo "   ./scripts/run_pipeline.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
cd backend
source .venv/bin/activate
cd ..

# Check database size for confirmation
DB_SIZE=$(du -h reports/enriched_instagram_data.sqlite | cut -f1)
echo "✅ Database found: $DB_SIZE"

# Start the server
echo ""
echo "🌐 Starting FastAPI server..."
echo "Dashboard URL: http://localhost:8000/"
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server from project root with proper path resolution
python backend/app/main.py 