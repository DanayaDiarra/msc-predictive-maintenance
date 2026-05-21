# Agentic Predictive Maintenance Platform

## 📋 Project Overview

This is an **Agentic AI for Predictive Maintenance** system for telecom infrastructure (BTS stations) in West Africa.
The project implements a multi-agent pipeline using XGBoost, PyTorch Transformers, and RAG for predictive maintenance.

**Main Application:** `orchestrai_noc.py` - OrchestrAI Network Operations Center (NOC) Dashboard

## 🚀 Quick Start

### Local Development

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Databases:**
   ```bash
   python seed_databases.py
   ```
   This creates three SQLite databases in `data/databases/`:
   - `hr_database.db` - Engineer roster (14 engineers)
   - `supply_chain.db` - Parts and inventory
   - `station_streams.db` - Telemetry and RUL predictions

3. **Run the Application:**
   ```bash
   streamlit run orchestrai_noc.py
   # OR
   streamlit run app.py
   # OR (Linux/Mac)
   ./START_APP.sh
   ```

4. **Access the Dashboard:**
   Open your browser to `http://localhost:8501`

## 🏗️ Project Structure

```
agentic_pdm/
├── orchestrai_noc.py          # Main Streamlit dashboard (use this!)
├── app.py                      # Entry point wrapper
├── START_APP.sh               # Startup script
│
├── config/
│   └── stations.py            # Centralized station configuration (15 BTS stations)
│
├── Core Agents (Production):
│   ├── interpreter_agent.py   # Alert construction
│   ├── diagnostic_agent.py    # Root cause analysis
│   └── planning_agent.py      # Action planning + execution
│
├── Orchestration:
│   ├── agentic_pdm_orchestrator.py  # Master pipeline runner
│   └── 00_setup_and_run.py          # Notebook setup helper
│
├── RAG Pipeline:
│   ├── rag_pipeline.py              # Retrieval logic
│   ├── rag_corpus_builder.py        # Knowledge base builder
│   └── rag_document_ingestor.py     # Document processing
│
├── Data Layer:
│   ├── db_connector.py              # Multi-DB connector (SQLite/PG/REST)
│   ├── data_connector.py            # Live sensor connector
│   ├── seed_databases.py            # Database initialization
│   └── ui_helpers.py                # Shared UI components
│
├── Model Training:
│   ├── train_model_phase1_2.py      # Model training
│   ├── retrain_pipeline.py          # Retraining workflow
│   └── realtime_feature_engine.py   # Real-time features
│
└── data/
    └── databases/                    # SQLite databases (auto-created)
```

## 🗄️ Database Configuration

The HR database is now automatically created and includes 14 engineers:
- **ENG001-ENG012:** Field engineers (power, thermal, RF, backhaul, baseband specialists)
- **MGR001-MGR002:** Managers

Database connection strings for Settings:
- **HR DB:** `SQLite · path: data/databases/hr_database.db`
- **Supply DB:** `SQLite · path: data/databases/supply_chain.db`
- **Streams DB:** `SQLite · path: data/databases/station_streams.db`

## 🎨 Features

- **Live Fleet Monitor** - Real-time station health and RUL predictions
- **Fleet Overview** - Fleet-wide metrics and urgency distribution
- **Station Detail** - Deep dive into individual station diagnostics
- **Dispatch & Roster** - Engineer management and dispatch planning
- **Engineer Chatbot** - AI-powered maintenance Q&A
- **Pipeline Intelligence** - Model metrics and ablation studies
- **Results & Ablation** - Performance analysis

## ☁️ Streamlit Cloud Deployment

### Manual Deployment Steps

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Fill in:
   - Repository: `DanayaDiarra/msc-predictive-maintenance`
   - Branch: `main`
   - Main file: `orchestrai_noc.py` or `app.py`
4. Click "Deploy"

### Set Environment Secrets
After deployment, go to app settings → Secrets and add:
```toml
USE_LLM = "true"
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

## 📝 Recent Updates (2026-05-21)

- ✅ Fixed HR database - now properly accessible with 14 engineers
- ✅ Removed code duplications (~3,200 lines removed)
- ✅ Centralized station configuration in `config/stations.py`
- ✅ Set `orchestrai_noc.py` as the primary dashboard
- ✅ Updated imports to use production agents
- ✅ Added startup script for easy launch

## 🛠️ Technologies

- **Frontend:** Streamlit
- **ML Models:** XGBoost, PyTorch Transformers
- **RAG:** Custom retrieval-augmented generation pipeline
- **Databases:** SQLite (development), PostgreSQL (production-ready)
- **AI:** Anthropic Claude API (optional, for LLM reasoning)

---

*Last updated: 2026-05-21*