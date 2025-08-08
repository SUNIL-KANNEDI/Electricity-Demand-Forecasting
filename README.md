# 🔌 Electricity Demand Forecasting for Kansai Electric Power (JP)

## 📘 Objective
To accurately forecast electricity demand using historical consumption and weather data through machine learning models: **Linear Regression**, **SVR**, and **XGBoost**.

---

## 📦 1. Data Loading & Preprocessing
**Source**: `dataset.csv` containing demand + weather data  
**Time Range**: Training till **2022-end**, testing starts post that  

**Features**:
- Demand
- Temperature
- Humidity
- Wind speed/direction
- Snowfall
- Location

**Preprocessing Steps**:
- Datetime parsing (`hour`, `dayofweek`, `month`)
- Lag features (`t-1`, `t-24`, `t-168`)
- Rolling mean (24-hour)
- Train-test split (time-based, no shuffle)

---

## 🛠 2. Models Developed

### 📈 Linear Regression
- Full preprocessing pipeline with **StandardScaler** and **OneHotEncoder**
- Achieved **R²: 0.9923**
- Tuned with **Ridge regularization** via `GridSearchCV`

### 🤖 Support Vector Regression (SVR)
- Used **SVR** with `rbf` kernel
- R² (on limited data due to SVR scale limitation): **0.9734**

### 🚀 XGBoost Regressor
- Most accurate model with **R²: 0.9945**
- Custom parameters: `max_depth=3`, `subsample=0.8`, `n_estimators=100`

---

## 📊 3. Evaluation Metrics

| Model             | MAE   | RMSE  | MAPE  |
|-------------------|-------|-------|-------|
| Linear Regression | 13.25 | 28.38 | 0.82% |
| SVR               | 23.25 | 52.86 | 1.37% |
| XGBoost           | 11.83 | 24.04 | 0.74% |

✅ **XGBoost** performed best in all metrics.

---

## 📉 4. Visualizations
- **Actual vs Predicted** plots for each model using `matplotlib`
- Time-series visualizations showing tight tracking of actual values (especially **XGBoost**)

---

## 💡 5. Hypotheses for Further Improvement
- Use **deep learning models** (LSTM, GRU, Transformer)
- Add **holiday/event indicators**
- Incorporate **external market data** (if available)
- Perform **multi-step ahead forecasting**

---

## 🚀 6. Deployment Benefits
- Better demand-side management
- Operational cost savings
- Reduced power outages
- Enhanced planning for peak vs off-peak hours

---

## 📂 Repository Structure
