import streamlit as st
import joblib
import numpy as np  # Added for validation

# Load the trained model and scaler
try:
    model = joblib.load('lr_model.sav')
    scaler = joblib.load('scaler_model.sav')
except Exception as e:
    st.error(f"Error loading the model or scaler: {e}")
    st.stop()

# Streamlit UI
st.title('Medical Expenditures Prediction')
st.write('Input the following features to get a prediction:')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100)
sex = st.selectbox('Gender', ['Male', 'Female'])
bmi = st.number_input('BMI', min_value=0.0, max_value=500.00)
children = st.number_input('Number of Children', min_value=0, max_value=100)
smoker = st.selectbox('Smoker', ['No', 'Yes'])
region = st.selectbox('Region', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

# Convert categorical inputs to numeric
sex_num = 1 if sex == 'Male' else 0
smoker_num = 1 if smoker == 'Yes' else 0

# Encode region using label encoding
region_encoded = [0, 1, 2, 3][['Northeast', 'Northwest', 'Southeast', 'Southwest'].index(region)]

# Prepare input features for prediction
input_features = [[age, sex_num, bmi, children, smoker_num, region_encoded]]

# Scale the input features
try:
    scaled_input_features = scaler.transform(input_features)
except Exception as e:
    st.error(f"Error scaling input features: {e}")
    st.stop()

# Predict button
if st.button('Predict'):
    prediction = model.predict(scaled_input_features)
    
    st.write(f'Predicted Medical Charges: ${"{:,.2f}".format(prediction[0])}')

