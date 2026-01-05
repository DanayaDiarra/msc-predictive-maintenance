# MSc Thesis: Predictive Maintenance Platform

**Thesis Title:** Agentic AI for Predictive Maintenance in Distributed Telecom Infrastructure

## 📋 Overview
A lightweight end-to-end platform combining time-series deep learning with LLM-based agent workflows for predictive maintenance.

## 🗓️ Timeline
4 months (January - April 2025)

## 📁 Repository Structure
<img width="659" height="226" alt="image" src="https://github.com/user-attachments/assets/01778aee-a047-49d5-8b83-3dc3cc1776e9" />

## 🚀 Getting Started
1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run exploration: `jupyter notebook notebooks/01_data_exploration.ipynb`

## 📊 Progress
- Week 1: Dataset exploration and baseline setup

## 📅 Week 1 Progress Report (Completed)

### ✅ **Objectives Achieved:**
1. **Repository Setup** - Complete project structure with SSH authentication
2. **Data Acquisition** - NASA FD001 dataset downloaded and verified
3. **Exploratory Data Analysis** - Comprehensive 10-step EDA completed
4. **Data Validation** - No missing values, clean structure confirmed
5. **Initial Insights** - Key findings documented for modeling

### 📊 **EDA Key Findings:**
- **Dataset:** 100 engines, 20,631 observations, 26 features
- **Quality:** Excellent - no missing values, consistent types
- **Lifetimes:** 128-362 cycles (mean: 206.3 ± 46.3)
- **Correlations:** sensor_09 ↔ sensor_14 highly correlated (r=0.963)
- **Anomalies:** Baseline detection shows ~2.37% anomaly rate

### 📁 **Week 1 Deliverables:**
- ✅ `notebooks/exploration/01_nasa_data_analysis.ipynb` - Complete EDA
- ✅ `docs/week1_eda_summary.md` - Detailed findings and insights
- ✅ `src/features/feature_engineering.py` - Week 2 implementation plan
- ✅ Repository updated with all code and documentation

### 🎯 **Ready for Week 2:**
The dataset is validated and ready for feature engineering and model development.

**Next:** Feature engineering pipeline → Baseline models → Deep learning implementation
