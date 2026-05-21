#!/bin/bash
# OrchestrAI NOC Startup Script
# ==============================
#
# This script starts the Agentic PdM Streamlit application
#
# Usage: ./START_APP.sh

echo "Starting OrchestrAI NOC..."
echo "================================"
echo ""
echo "Main application: orchestrai_noc.py"
echo "Database location: data/databases/"
echo ""

# Check if databases exist
if [ ! -d "data/databases" ]; then
    echo "⚠️  Database directory not found. Creating and seeding databases..."
    python seed_databases.py
fi

echo "🚀 Launching Streamlit app..."
streamlit run orchestrai_noc.py
