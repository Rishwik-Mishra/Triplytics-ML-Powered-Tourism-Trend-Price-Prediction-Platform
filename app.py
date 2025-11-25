import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import re  # Import regular expressions for cleaning

# --- 1. SET PAGE CONFIG (Must be first Streamlit command) ---
st.set_page_config(page_title="Triplytics", page_icon="💰", layout="wide")


# --- 2. DEFINE STATIC DATA (Maps for Flight Model) ---
# These are the manual mappings for the FLIGHT model
airline_map = {
    'Vistara': 1, 'Air_India': 2, 'Indigo': 3,
    'GO_FIRST': 4, 'AirAsia': 5, 'SpiceJet': 6
}
source_city_map = {
    'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3,
    'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6
}
departure_time_map = {
    'Morning': 1, 'Early_Morning': 2, 'Evening': 3,
    'Night': 4, 'Afternoon': 5, 'Late_Night': 6
}
stops_map = {
    'one': 1, 'zero': 2, 'two_or_more': 3
}
destination_city_map = {
    'Mumbai': 1, 'Delhi': 2, 'Bangalore': 3,
    'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6
}
flight_class_map = {
    'Economy': 1, 'Business': 2
}


# --- 3. OPTIMIZED MODEL & DATA LOADING ---
@st.cache_resource
def load_data_and_models():
    """
    Loads all models, preprocessors, and supplementary data.
    Implements the Top 125 + Cascading Dropdown logic.
    Returns: A tuple containing all loaded objects.
    """
    
    def normalize_name(name):
        """Helper function to clean station codes and names."""
        if not isinstance(name, str):
            return ""
        name = name.upper().strip()
        name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with one
        return name

    try:
        # Load flight models
        flight_model = joblib.load('flight_model.pkl')
        flight_scaler = joblib.load('scaler.pkl')
        
        # Load train model
        train_pipeline = joblib.load('train_price_pipeline.pkl')
        
        # Load and CLEAN Station Name Mapping
        station_map_file = 'list_of_stations.json' 
        with open(station_map_file, 'r') as f:
            station_data = json.load(f)
        
        code_to_name_map = {}
        name_to_code_map = {}
        for item in station_data:
            code = normalize_name(item.get('station_code'))
            name = normalize_name(item.get('station_name'))
            if code and name:
                if code not in code_to_name_map:
                    code_to_name_map[code] = name
                if name not in name_to_code_map:
                    name_to_code_map[name] = code

        # Load and CLEAN the expanded train data file
        df_train_data_raw = pd.read_csv('passenger_train_data_expanded.csv', sep=',')
        df_train_data_raw['fromStnCode'] = df_train_data_raw['fromStnCode'].apply(normalize_name)
        df_train_data_raw['toStnCode'] = df_train_data_raw['toStnCode'].apply(normalize_name)
        
        # 1. Clean the data (drop NaNs from critical columns)
        required_cols = ['fromStnCode', 'toStnCode', 'distance', 'duration']
        df_train_data = df_train_data_raw.dropna(subset=required_cols)

        # 2. Create the main route lookup (for distance/duration)
        route_lookup = df_train_data.groupby(
            ['fromStnCode', 'toStnCode']
        )[['distance', 'duration']].first().to_dict('index')

        # --- THIS IS THE COMBINED LOGIC ---
        
        # 3. Find the Top 125 station codes from the *clean* data
        from_counts = df_train_data['fromStnCode'].value_counts()
        to_counts = df_train_data['toStnCode'].value_counts()
        all_station_counts = from_counts.add(to_counts, fill_value=0).sort_values(ascending=False)
        top_125_codes = set(all_station_counts.head(125).index.tolist()) # Use a set for fast lookups

        # 4. Create the dependency map (fromStnCode -> [list of toStnCodes])
        source_to_dest_map = {}
        for source_code, group in df_train_data.groupby('fromStnCode'):
            # Only consider this source if it's in our Top 125 list
            if source_code not in top_125_codes:
                continue

            source_name = code_to_name_map.get(source_code)
            if not source_name: # Skip if Top 125 code has no matching name
                continue
            
            dest_codes = group['toStnCode'].unique()
            dest_names = sorted(list(set(
                code_to_name_map[code] for code in dest_codes
                if code in code_to_name_map # Ensure destination also has a name
            )))
            
            # Only add the source if it has valid, named destinations
            if dest_names:
                source_to_dest_map[source_name] = dest_names
        
        # 5. Create the SOURCE dropdown list (now only Top 125 Hubs)
        source_name_list = sorted(list(source_to_dest_map.keys()))
        
        # --- END OF COMBINED LOGIC ---

        return (flight_model, flight_scaler, train_pipeline, 
                source_name_list, source_to_dest_map, 
                name_to_code_map, route_lookup)

    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}. Make sure all .pkl, .csv, and .json files are in the same folder as app.py.")
        return None, None, None, [], {}, {}, {}
    except Exception as e:
        st.error(f"An error occurred during loading: {e}")
        return None, None, None, [], {}, {}, {}

# Load all models and data at startup
(flight_model, flight_scaler, train_pipeline, 
 source_name_list, source_to_dest_map, 
 name_to_code_map, route_lookup) = load_data_and_models()


# --- 4. FLIGHT PREDICTOR UI FUNCTION ---
def show_flight_predictor():
    st.subheader("Predict Flight Prices ✈️")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Flight Route')
        airline = st.selectbox('Airline', options=list(airline_map.keys()), key='flight_airline')
        source_city = st.selectbox('Source City', options=list(source_city_map.keys()), key='flight_source')
        destination_city = st.selectbox('Destination City', options=list(destination_city_map.keys()), key='flight_dest')
        flight_class = st.selectbox('Class', options=list(flight_class_map.keys()), key='flight_class')

    with col2:
        st.subheader('Flight Details')
        departure_time = st.selectbox('Departure Time', options=list(departure_time_map.keys()), key='flight_dep_time')
        stops = st.selectbox('Stops', options=list(stops_map.keys()), key='flight_stops')
        days_left = st.number_input('Days Left Until Departure', min_value=1, max_value=50, step=1, value=15, key='flight_days')

    st.divider()
    if st.button('Predict Flight Price', type="primary", use_container_width=True):
        if flight_model is None or flight_scaler is None:
            st.error("Flight model is not loaded. Please check file paths.")
            return

        # 1. Convert text inputs to their mapped numbers
        airline_num = airline_map[airline]
        source_city_num = source_city_map[source_city]
        departure_time_num = departure_time_map[departure_time]
        stops_num = stops_map[stops]
        destination_city_num = destination_city_map[destination_city]
        flight_class_num = flight_class_map[flight_class]

        # 2. Create the input DataFrame
        input_data = pd.DataFrame({
            'airline': [airline_num],
            'source_city': [source_city_num],
            'destination_city': [destination_city_num],
            'departure_time': [departure_time_num],
            'stops': [stops_num],
            'flight_class': [flight_class_num],
            'days_left': [days_left]
        })
        
        try:
            # 3. Scale the input data and restore feature names
            input_scaled_array = flight_scaler.transform(input_data)
            input_scaled_df = pd.DataFrame(input_scaled_array, columns=input_data.columns)
            
            # 4. Make the prediction
            prediction = flight_model.predict(input_scaled_df)
            predicted_price = np.round(prediction[0], 2)

            # 5. Display the result
            st.success(f'The predicted flight price is: ₹ {predicted_price}')

        except Exception as e:
            st.error(f"An error occurred during flight prediction: {e}")


# --- 5. TRAIN PREDICTOR UI FUNCTION (Corrected) ---
def show_train_predictor():
    st.subheader("Predict Train Prices 🚆")
    
    if not source_name_list or not source_to_dest_map:
        st.error("Station data is empty or failed to load. Check if 'passenger_train_data_expanded.csv' and 'list_of_stations.json' are loaded correctly.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Route & Class')
        
        # --- SOURCE DROPDOWN IS NOW ONLY TOP 125 HUBS ---
        from_stn_name = st.selectbox('Source Station (Top 125 Hubs)', options=source_name_list, key='train_source_name')
        
        # --- Dependent Destination Dropdown ---
        available_destinations = []
        if from_stn_name in source_to_dest_map:
            available_destinations = source_to_dest_map[from_stn_name]
        
        to_stn_name = st.selectbox('Destination Station', options=available_destinations, key='train_dest_name')
        
        class_code = st.selectbox('Train Class', 
                                  options=['3A', 'SL', '2A', '1A', '3E', '2S', 'CC', 'FC'], 
                                  key='train_class')
    
    with col2:
        st.subheader('Booking Details')
        booking_month = st.slider('Booking Month', min_value=1, max_value=12, value=10, key='train_month')
        booking_day = st.slider('Booking Day of Week (0=Mon, 6=Sun)', min_value=0, max_value=6, value=1, key='train_day')
        booking_hour = st.slider('Booking Hour (0-23)', min_value=0, max_value=23, value=22, key='train_hour') 

    st.divider()
    if st.button('Predict Train Price', type="primary", use_container_width=True):
        if train_pipeline is None or not route_lookup or not name_to_code_map:
            st.error("Train model or data is not loaded. Please check file paths.")
            return

        # Handle case where no destination is selected
        if not to_stn_name:
            st.error("Please select a destination.")
            return

        # Convert selected names back to codes
        from_stn_code = name_to_code_map[from_stn_name]
        to_stn_code = name_to_code_map[to_stn_name]
        route_key = (from_stn_code, to_stn_code)
        
        # This check should now be much more reliable
        if route_key not in route_lookup:
            st.error(f"Sorry, no price data found for the direct route {from_stn_name} to {to_stn_name}.")
            st.info(f"Debug: Route key ({from_stn_code}, {to_stn_code}) not in route_lookup.")
            return
        
        # Get the distance and duration from our lookup
        route_data = route_lookup[route_key]
        distance = route_data['distance']
        duration = route_data['duration']

        # 1. Create the input DataFrame (using codes, as the model expects)
        input_data = pd.DataFrame({
            'fromStnCode': [from_stn_code],
            'toStnCode': [to_stn_code],
            'classCode': [class_code],
            'distance': [distance],
            'duration': [duration],
            'booking_month': [booking_month],
            'booking_day_of_week': [booking_day],
            'booking_hour': [booking_hour]
        })

        try:
            # 2. Make the prediction (pipeline handles all preprocessing)
            prediction = train_pipeline.predict(input_data)
            predicted_price = np.round(prediction[0], 2)

            # 3. Display the result
            st.success(f'The predicted train price is: ₹ {predicted_price}')
            st.caption(f"Based on a route from {from_stn_code} to {to_stn_code} (Distance: {distance} km, Duration: {duration} min).")
        
        except Exception as e:
            st.error(f"An error occurred during train prediction: {e}")


# --- 6. MAIN APP LOGIC ---
st.title("📈 Triplytics: Tourism Trend Analysis and Predictive Price Modeling")

predictor_choice = st.radio(
    "Select a predictor:",
    ('Flight Price Predictor ✈️', 'Train Price Predictor 🚆'),
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

if predictor_choice == 'Flight Price Predictor ✈️':
    show_flight_predictor()
else:
    show_train_predictor()