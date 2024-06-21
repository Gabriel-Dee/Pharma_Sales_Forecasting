import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Function to preprocess data for VAR model
@st.cache_data
def preprocess_data(data):
    df_encoded = pd.get_dummies(data, columns=['Pharmacy Name', 'Product Code'])
    
    features = df_encoded.drop(columns=['Sales', 'Date'])
    target = df_encoded['Sales']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    train_size = int(len(df_encoded) * 0.8)
    train_features, test_features = features_scaled[:train_size], features_scaled[train_size:]
    train_target, test_target = target[:train_size], target[train_size:]
    
    train_target = train_target.reset_index(drop=True)
    train_features = pd.DataFrame(train_features, columns=features.columns)
    
    var_data = pd.concat([train_target, train_features], axis=1)
    last_date = data['Date'].iloc[train_size - 1]  # Get the last date in training data
    return var_data, last_date

# Function to fit the VAR model and make forecasts
def fit_var_model(data, lag_order, forecast_steps, last_date):
    model = VAR(data)
    var_result = model.fit(lag_order)
    
    last_obs = data.values[-lag_order:]
    var_forecast = var_result.forecast(y=last_obs, steps=forecast_steps)
    
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
    forecast_sales = var_forecast[:, 0]  # Select only the forecasted sales column
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted_Sales': forecast_sales
    })
    
    return forecast_df

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')
lag_order = st.sidebar.number_input('Select the lag order ( Including more lagged values allows the model to capture longer-term dependencies and patterns in the data. ):', min_value=1, max_value=15, value=5, step=1)
forecast_steps = st.sidebar.number_input('Number of periods to forecast:', min_value=1, max_value=365, value=6, step=1)

# Title
st.title('Sales Forecasting Using VAR Model')

# Load data
data_path = 'Data/data.csv'
data = load_data(data_path)

# Preprocess data
var_data, last_date = preprocess_data(data)

# Fit the VAR model and make forecasts
if st.button('Forecast'):
    forecast_df = fit_var_model(var_data, lag_order, forecast_steps, last_date)
    
    # Display forecast data
    st.subheader('Forecast Data')
    st.write(forecast_df)
    
    # Plot forecasted sales
    st.subheader('Forecast Plot')
    st.line_chart(forecast_df.set_index('Date')['Forecasted_Sales'])

