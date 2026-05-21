# OrchestrAI Network Operations Center

**Advanced AI-Powered Predictive Maintenance Platform for Telecommunications Infrastructure**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://msc-predictive-maintenance-kuviqvk7aflphb8zjbidhp.streamlit.app/)

## 📋 Overview

OrchestrAI NOC is an enterprise-grade **agentic AI system** for predictive maintenance of telecommunications infrastructure across West Africa. The platform combines multi-agent reasoning, real-time monitoring, and intelligent dispatch planning to predict equipment failures and optimize maintenance operations.

**Live Demo:** [https://msc-predictive-maintenance-kuviqvk7aflphb8zjbidhp.streamlit.app/](https://msc-predictive-maintenance-kuviqvk7aflphb8zjbidhp.streamlit.app/)

### Key Features

- 🤖 **Multi-Agent AI Pipeline** - Interpreter, Diagnostic, and Planning agents with RAG knowledge base
- 📊 **Real-Time Monitoring** - Live tracking of 25 BTS stations with RUL prediction (RMSE: 15.11 cycles)
- 🚨 **Intelligent Dispatch** - Automated engineer assignment from 27-person roster with skill matching
- 🗄️ **Integrated Data Sources** - HR database, supply chain inventory, and telemetry streams
- 💬 **AI Chatbot** - Multi-tier fallback system (Groq LLaMA 3.3 70B → Claude Haiku → Rule-based KB)
- 📱 **SMS Notifications** - Automatic alerts to dispatched engineers

**Author:** Danaya Diarra
**Institution:** Graduate School of Management, Saint Petersburg State University
**Year:** 2026

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
   - `hr_database.db` - Engineer roster (27 engineers: 13 Mali, 14 Senegal)
   - `supply_chain.db` - Parts and inventory
   - `station_streams.db` - Telemetry and RUL predictions for 25 stations

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
│   └── stations.py            # Centralized station configuration (25 BTS stations)
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

The HR database is now automatically created and includes 27 engineers:
- **ENG001-ENG027:** Field engineers (power, thermal, RF, backhaul, baseband specialists)
  - 8 Senior, 10 Mid-Level, 9 Junior engineers
  - 13 based in Mali, 14 based in Senegal
  - All with phone numbers (+221/+223) and shift assignments (16 Day, 11 Night)
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

- ✅ Scaled to 25 stations (12 Mali, 11 Senegal, 2 others) and 27 engineers
- ✅ Added phone notifications and shift management (Day/Night) for engineers
- ✅ Fixed HR database - now properly accessible with 27 engineers
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