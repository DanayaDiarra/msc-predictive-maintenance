# Agentic Predictive Maintenance Platform

## 🚀 Streamlit Cloud Deployment

### 1. One-Click Deploy
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

### 2. Manual Deployment Steps

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Fill in:
   - Repository: `DanayaDiarra/msc-predictive-maintenance`
   - Branch: `main`
   - Main file: `streamlit_dashboard_v2.py`
4. Click "Deploy"

### 3. Set Environment Secrets
After deployment, go to app settings → Secrets and add:
```toml
USE_LLM = "true"
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

### 4. Access Your App
`https://danayadiarra-msc-predictive-maintenance-streamlit-dashboard-v2.streamlit.app`


*Last updated: 2026-04-03 00:39:37*