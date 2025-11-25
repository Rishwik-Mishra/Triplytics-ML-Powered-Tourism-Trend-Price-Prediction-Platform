📈 Triplytics: ML-Powered Tourism Trend & Price Prediction Platform

 Role: Wroked in a Team of two | Status: Active

🚀 Overview

Triplytics is a comprehensive predictive analytics system designed to forecast travel costs for both flights and trains with high precision. Unlike standard fare calculators, Triplytics uses machine learning to predict future prices based on temporal booking trends, route popularity, and dynamic pricing factors.

This project processes over 300,000+ records of flight and rail data to train robust regression models, offering users a unified interface to estimate their travel expenses days or weeks in advance.

🎯 Key Features

Dual-Mode Prediction Engine: A unified hub for predicting both Flight and Train ticket prices.

Cascading Logic System: Implemented smart, interdependent dropdowns for train stations. The destination list dynamically filters based on the selected source station, ensuring 100% valid route queries.

Hub-and-Spoke Route Modeling: The train model is trained on a curated dataset of 125+ major railway hubs, covering thousands of high-traffic routes across the network.

Temporal Feature Engineering: extracting deep insights from booking timestamps (e.g., booking day, hours until departure) to capture dynamic pricing surges.

Interactive Dashboard: A responsive, user-friendly web app built with Streamlit, featuring real-time inference and clear visualizations.

🛠️ Technical Architecture

1. Machine Learning Pipeline

Algorithms: Utilized Random Forest Regressor (n_estimators=100) for its ability to handle non-linear relationships and interaction effects between features (e.g., Airline × Departure Time).

Data Scale:

Flight Data: ~300,000 records (clean Kaggle dataset).

Train Data: ~180,000 records (custom filtered dataset focusing on major hubs).

Preprocessing:

One-Hot Encoding: For categorical variables like Airline, Source, Destination, and Class.

Standard Scaling: Applied to numerical features like Distance and Duration to ensure model stability.

Pipelines: Used sklearn.pipeline to bundle preprocessing and modeling, preventing data leakage and simplifying inference.

2. Data Engineering

Data Cleaning: Engineered a robust cleaning script to normalize inconsistent station names (e.g., "KANPUR CENTRAL JN." vs "Kanpur Central").

Feature Extraction: Derived critical features such as days_left, journey_day_of_week, and booking_hour from raw timestamps.

Optimization: Implemented a Route Lookup Dictionary (Hash Map) for O(1) retrieval of route distances and durations, significantly reducing inference latency.

3. Frontend (Streamlit)

State Management: Used Streamlit's session state and caching (@st.cache_resource) to load heavy ML models only once, reducing app load time by 90% on subsequent runs.

Dynamic UI: Designed a conditional rendering system that swaps input forms based on the user's mode selection (Flight vs. Train).

📊 Model Performance

Metric

Flight Model

Train Model

R² Score

0.98

0.99

MAE

Low error margin

Precision within ~₹50

The high R² scores demonstrate the models' strong ability to explain price variance based on the selected features.

💻 Installation & Setup

Clone the repository:

git clone [https://github.com/Rishwik-Mishra/Triplytics-ML-Powered-Tourism-Trend-Price-Prediction-Platform.git](https://github.com/Rishwik-Mishra/Triplytics-ML-Powered-Tourism-Trend-Price-Prediction-Platform.git)
cd Triplytics


Install dependencies:

pip install -r requirements.txt


Run the application:

streamlit run app.py


📂 Project Structure

Triplytics/
├── app.py                       # Main Streamlit application entry point
├── flight_model.pkl             # Trained Random Forest model for flights
├── train_price_pipeline.pkl     # Trained ML pipeline for trains
├── scaler.pkl                   # Feature scaler for normalization
├── passenger_train_data_expanded.csv  # Processed train dataset (180k+ rows)
├── list_of_stations.json        # Station code-to-name mapping
├── Price_Prediction_Flight.ipynb # Jupyter notebook for flight model training
└── train_price_predictor.ipynb   # Jupyter notebook for train model training


👨‍💻 Author

Rishwik Mishra

Role: Full Stack Data Science Developer

Focus: Machine Learning, Data Analytics, Web Development

*Star ⭐ this repository
