import joblib
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Page title
st.set_page_config(page_title='Sucide Rate Prediction')
st.title('Sucide Rate Prediction')
st.info('This is a Sucide Rate prediction mainly based in KL & Selangor area. Fill in the information and check the predicted price below.')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_data.csv').copy()  # Make a copy of the DataFrame to avoid mutation

# Define region bins
region_bins = {
    'Africa': 'Africa',
    'Asia': 'Asia',
    'Central South America': 'Central_South_America',
    'Europe': 'Europe',
    'North America Caribbean': 'North_America_Caribbean',
    'Oceania': 'Oceania'
}

# Define Sex bins
sex_bins = {
    'Female': 0,
    'Male': 1,
    'Unknown': 2
}

# Define location bins
generation_bins = {
    'Unknown': 0,
    'Silent Generation': 1,
    'Baby Boomers': 2,
    'Generation X': 3,
    'Millennials': 4,
    'Generation Z': 5,
    'Generation Alpha': 6
}

data = load_data()
label_encoder = LabelEncoder()
label_encoder.fit(data['RegionName'])

# Train the model
model = joblib.load('suicide_count_prediction_model_RF.pkl')

# Sucide Rate prediction form
with st.form('predict'):
    RegionName = st.selectbox('Region', list(region_bins.keys()))
    Generation = st.selectbox('Generation', list(generation_bins.keys()))
    Sex = st.selectbox('Sex',list(sex_bins.keys()))
    Population = st.number_input('Population')
    GDP = st.number_input('GDP')
    GrossNationalIncome = st.number_input('Gross National Income')
    InflationRate = st.number_input('Inflation Rate')
    EmploymentPopulationRatio = st.number_input('Employment Population Ratio')
    submit = st.form_submit_button('Predict')

if submit:
    RegionName = int(label_encoder.transform([RegionName])[0])
    Generation = generation_bins[Generation]
    Sex = sex_bins[Sex]
    RegionName = region_bins[RegionName]
    input_data = [[RegionName, Generation, Sex, Population, GDP, GrossNationalIncome, InflationRate, EmploymentPopulationRatio]]
    prediction = model.predict(input_data)
    st.write("Predicted Sucide Rate:", round(prediction[0], 2))
