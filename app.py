import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)          



st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
gender_encoder = label_encoder_gender.transform([gender])[0]
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=1)
num_of_products = st.number_input("Number of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Gender': [gender_encoder],
    'Geography_France': [1 if geography=="France" else 0],
    'Geography_Germany': [1 if geography=="Germany" else 0],
    'Geography_Spain': [1 if geography=="Spain" else 0]
})

geography_encoder = onehot_encoder_geography.transform([[geography]]).toarray()
geography_encoder_df = pd.DataFrame(geography_encoder, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))




input_data = pd.concat([pd.DataFrame(input_data), geography_encoder_df], axis=1)
input_data= input_data.loc[:, ~input_data.columns.duplicated()]

input_data = input_data[scaler.feature_names_in_]
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probe = prediction[0][0]

if prediction_probe > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")