# 📈 Triplytics: ML-Powered Tourism Trend & Price Prediction Platform

**Role:** Worked in a Team of Two  
**Status:** Active

---

## 🚀 Overview

Triplytics is a comprehensive predictive analytics system designed to **forecast travel costs for both flights and trains with high precision**. Unlike standard fare calculators, Triplytics leverages machine learning to provide real-time and future price predictions based on extensive historical data, robust preprocessing, and sophisticated feature engineering.

This project processes **over 300,000+ records** of flight and rail data to train advanced regression models, offering users a **unified interface to estimate travel expenses days or weeks in advance**.

---

## 🎯 Key Features

- **Dual-Mode Prediction Engine**  
  One platform to predict both **Flight** and **Train** ticket prices.

- **Cascading Logic System**  
  Intuitive dropdowns for train stations: the destination list automatically updates based on the chosen source, ensuring only valid routes.

- **Hub-and-Spoke Route Modeling**  
  The train pricing model covers **125+ major railway hubs** and thousands of high-traffic interconnections.

- **Temporal Feature Engineering**  
  Smart extraction from booking timestamps (e.g., advance days, booking day of week, booking hour) to capture price surges and trends.

- **Interactive Dashboard**  
  Built with **Streamlit**, featuring real-time inference and clear price visualizations.

---

## 🛠️ Technical Architecture

### 1. Machine Learning Pipeline

- **Algorithm:**  
  Random Forest Regressor (`n_estimators=100`) for handling complex, non-linear price dynamics.

- **Data Used:**  
  - **Flight Data:** ~300,000 records (cleaned Kaggle dataset)  
  - **Train Data:** ~180,000 records (custom-filtered for major hubs)

- **Preprocessing Flow:**  
  - **One-Hot Encoding:** Airline, Source, Destination, Class  
  - **Standard Scaling:** Distance, Duration  
  - **Pipelines:** All steps managed via `sklearn.pipeline` to prevent data leakage and ensure replicable inference.

---

### 2. Data Engineering

- **Data Cleaning:**  
  Normalized inconsistent station names (e.g., `"KANPUR CENTRAL JN."` ➔ `"Kanpur Central"`).

- **Feature Extraction:**  
  Derived features such as `days_left`, `journey_day_of_week`, `booking_hour` from raw timestamps.

- **Optimization:**  
  Built a **Route Lookup Dictionary** (Hash Map) for instant retrieval of route details, drastically reducing prediction latency.

---

### 3. Frontend (Streamlit)

- **State Management:**  
  Used Streamlit's session state and advanced caching (`@st.cache_resource`) to ensure fast loading (90% faster reloads).

- **Dynamic UI:**  
  Conditional rendering enables seamless toggling between **Flight** and **Train** modes with intuitive forms.

---

## 📊 Model Performance

| Metric     | Flight Model | Train Model       |
| ---------- | ------------ | ---------------- |
| **R² Score**  | 0.98         | 0.99             |
| **MAE**       | Low error    | Precision within ~₹50 |

High R² scores prove the models' strength in capturing fare variation.

---

## 💻 Installation & Setup

**1. Clone the repository:**
```sh
git clone https://github.com/Rishwik-Mishra/Triplytics-ML-Powered-Tourism-Trend-Price-Prediction-Platform.git
cd Triplytics-ML-Powered-Tourism-Trend-Price-Prediction-Platform
```

**2. Install dependencies:**
```sh
pip install -r requirements.txt
```

**3. Run the application:**
```sh
streamlit run app.py
```

---

## 📂 Project Structure

```
Triplytics/
├── app.py                           # Main Streamlit app
├── flight_model.pkl                 # Trained random forest model (flights)
├── train_price_pipeline.pkl         # Trained ML pipeline (trains)
├── scaler.pkl                       # Feature scaler for normalization
├── passenger_train_data_expanded.csv# Processed train dataset (180k+ rows)
├── list_of_stations.json            # Station code-to-name mapping
├── Price_Prediction_Flight.ipynb    # Flight model training notebook
└── train_price_predictor.ipynb      # Train model training notebook
```

---

## 👨‍💻 Author

**Rishwik Mishra**  
*Full Stack Data Science Developer*  
**Focus:** Machine Learning, Data Analytics, Web Development

---

⭐️ *Star this repository if you found it useful!*
