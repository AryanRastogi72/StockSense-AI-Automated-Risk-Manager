<div align="center">

# 📈 Stock-Ticker with ML and DL
### A Comparative Predictive Analysis

*An L&T Technology Services Internship Project*

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

</div>

---

## 🧭 Table of Contents

- #-Project-Overview[https://github.com/Saurov-Thakur_ltts/Intern_ml/edit/main/ReadME.md/Project Overview]
- #-Objective
- #-Repository-Contents
- #-Documentation
- #️-Implementations
- #-Models-Being-Compared 

## Project Overview

This project builds a **mobile-app-ready stock trend prediction system** that forecasts a stock's short-term price movement (**1–2 days ahead**), using **LTTS (L&T Technology Services, NSE: LTTS)** as the primary running example.

Instead of committing to a single algorithm upfront, this project **implements and evaluates multiple ML and DL approaches** on the same LTTS price data — under the same evaluation criteria — to empirically determine which approach performs best for short-term stock prediction.

---

## 🎯 Objective

> Build a **comparative predictive framework** that:
>
> - Cleans and prepares real NSE stock data
> - Engineers meaningful technical indicators
> - Trains multiple ML/DL models on the same dataset
> - Evaluates them fairly against a naive baseline
> - Wraps the best-performing model behind an API for mobile-app consumption

---

## 📂 Repository Contents

| Section | Description |
|---------|-------------|
| 📘 **Documentation** | Full project reference document covering all conceptual, statistical, and algorithmic groundwork |
| 🛠️ **Implementations** | All working code — feature engineering notebooks and model scripts |

> **Note:** Raw and processed datasets are excluded from the repository. Users should place their own NSE CSV file inside the `Implementations/` folder before running the pipeline.

---

## 📘 Documentation

The `Documentation/` folder contains the complete reference document for the project.

| File | Description |
|------|-------------|
| `Project_Reference_Document.docx` | Complete reference covering data types, statistical foundations, ML vs DL, technical indicators, algorithm comparisons, and the full logical pipeline |

**Highlights inside the document:**

- 🧩 Data Type & Category Tree (Structured / Semi-structured / Unstructured)
- 🧠 AI vs ML vs Deep Learning
- 📊 Statistical Foundations (Univariate, Multivariate, Overfitting/Underfitting)
- 📅 Time-Series Concepts & Industry Use Cases
- 🧪 Technical Indicators (MA, EMA, RSI, MACD, ROC, Bollinger Bands, OBV)
- 🤖 ML vs DL Comparison
- 🏗️ Logical Approach & Implementation Steps

---

## 🛠️ Implementations

The `Implementations/` folder contains the working code for feature engineering and model building.

| File | Purpose |
|------|---------|
| `Feature_Engineering.ipynb` | Loads the raw NSE CSV, performs cleaning, and generates all technical indicators listed in the reference document |
| `Linear_Regression_Model.py` | Trains a Linear Regression model on the engineered features and predicts the next day's closing price |

> Additional models (Logistic Regression, ARIMA, Random Forest, LSTM, GRU, Transformer) will be added in later phases of the project.

---

## 📊 Models Being Compared

| Category | Model | Type | Status |
|----------|-------|------|--------|
| Traditional ML | Moving Average | Baseline | 🕓 Planned |
| Traditional ML | **Linear Regression** | Regression | ✅ In Progress |
| Traditional ML | **Logistic Regression** | Classification | 🕓 Planned |
| Traditional ML | ARIMA | Time-Series | 🕓 Planned |
| Traditional ML | Random Forest | Ensemble | 🕓 Planned |
| Deep Learning | LSTM | Sequence Model | 🕓 Planned |
| Deep Learning | GRU | Sequence Model | 🕓 Planned |
| Deep Learning | Transformer | Attention Model | 🕓 Planned |

---

## 🧪 Technical Indicators Used

| Indicator | Category | Purpose |
|-----------|----------|---------|
| Lag Prices | Historical | Captures recent price history |
| Daily Return | Momentum | Direction + strength of change |
| Moving Average (MA5 / MA10 / MA20) | Trend | Smooths noise |
| Volatility | Risk | Measures instability |
| Log Volume | Volume | Scales large volume swings |
| RSI | Momentum | Buying vs selling exhaustion |
| MACD | Momentum | Trend momentum shifts |
| ROC | Momentum | Speed of price change |
| Bollinger Bands | Volatility | Unusual price stretch |
| OBV | Volume | Volume-backed conviction |

---

## 📈 Roadmap

- [x] Week 1 — Conceptual Foundations
- [x] Week 2 — Logical Approach & Technical Indicators
- [x] Feature Engineering Pipeline
- [ ] Linear Regression Model (In Progress)
- [ ] Logistic Regression Model
- [ ] ARIMA / Random Forest
- [ ] LSTM / GRU / Transformer
- [ ] Model Evaluation & Comparison
- [ ] API + Mobile App Integration

---

## 👤 Author

**Aryan Rastogi**  
*Intern @ L&T Technology Services*  
🏢 Bangalore, India  
🧑‍💼 Supervisor: Saurov Thakur

---

<div align="center">

*Built with ☕, curiosity, and a lot of `df.head()` calls.*

</div>
