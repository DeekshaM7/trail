import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load your dataset (uncomment and add your data loading logic)
df = pd.read_csv("Finaldf-2.csv")

# Load your trained models from pickle files
with open('knn_model.pkl', 'rb') as knn_file:
    knn_model = pickle.load(knn_file)

with open('naive_bayes_model.pkl', 'rb') as nb_file:
    nb_model = pickle.load(nb_file)

with open('logistic_regression_model.pkl', 'rb') as lr_file:
    lr_model = pickle.load(lr_file)

with open('random_forest_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

# Set Streamlit page title and icon
st.set_page_config(page_title="Fraud Detection App", page_icon="✅")

# Define a function for making predictions with a selected model
def predict_with_model(model, data):
    prediction = model.predict(data)
    return prediction

# Define functions for each page
def knn_page():
    st.title("K-Nearest Neighbors (KNN) Page")

    # Example: User inputs data for prediction
    cc_num = st.text_input("Credit Card Number")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population")
    gender_enc = st.number_input("Gender (0 or 1)")
    state_enc = st.number_input("State Code")
    street_enc = st.number_input("Street Code")
    city_enc = st.number_input("City Code")
    job_enc = st.number_input("Job Code")
    category_enc = st.number_input("Category Code")

    # Create a feature vector from the user input
    user_input = np.array([cc_num, amt, lat, long, city_pop, gender_enc, state_enc, street_enc, city_enc, job_enc, category_enc]).reshape(1, -1)

    if st.button("Predict"):
        prediction = predict_with_model(knn_model, user_input)
        st.write("Prediction:", prediction[0])

def nb_page():
    st.title("Naive Bayes Page")

    # Example: User inputs data for prediction
    cc_num = st.text_input("Credit Card Number")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population")
    gender_enc = st.number_input("Gender (0 or 1)")
    state_enc = st.number_input("State Code")
    street_enc = st.number_input("Street Code")
    city_enc = st.number_input("City Code")
    job_enc = st.number_input("Job Code")
    category_enc = st.number_input("Category Code")

    # Create a feature vector from the user input
    user_input = np.array([cc_num, amt, lat, long, city_pop, gender_enc, state_enc, street_enc, city_enc, job_enc, category_enc]).reshape(1, -1)

    if st.button("Predict"):
        prediction = predict_with_model(nb_model, user_input)
        st.write("Prediction:", prediction[0])

def logistic_page():
    st.title("Logistic Regression Page")

    # Example: User inputs data for prediction
    cc_num = st.text_input("Credit Card Number")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population")
    gender_enc = st.number_input("Gender (0 or 1)")
    state_enc = st.number_input("State Code")
    street_enc = st.number_input("Street Code")
    city_enc = st.number_input("City Code")
    job_enc = st.number_input("Job Code")
    category_enc = st.number_input("Category Code")

    # Create a feature vector from the user input
    user_input = np.array([cc_num, amt, lat, long, city_pop, gender_enc, state_enc, street_enc, city_enc, job_enc, category_enc]).reshape(1, -1)

    if st.button("Predict"):
        prediction = predict_with_model(lr_model, user_input)
        st.write("Prediction:", prediction[0])

def rf_page():
    st.title("Random Forest Classifier Page")

    # Example: User inputs data for prediction
    cc_num = st.text_input("Credit Card Number")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population")
    gender_enc = st.number_input("Gender (0 or 1)")
    state_enc = st.number_input("State Code")
    street_enc = st.number_input("Street Code")
    city_enc = st.number_input("City Code")
    job_enc = st.number_input("Job Code")
    category_enc = st.number_input("Category Code")

    # Create a feature vector from the user input
    user_input = np.array([cc_num, amt, lat, long, city_pop, gender_enc, state_enc, street_enc, city_enc, job_enc, category_enc]).reshape(1, -1)

    if st.button("Predict"):
        prediction = predict_with_model(rf_model, user_input)
        st.write("Prediction:", prediction[0])

def eda_page():
    st.title("Exploratory Data Analysis (EDA) Page")
    # Your EDA code, e.g., use Plotly or other visualization libraries

# Create a sidebar navigation menu
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox(
    "Choose a page:",
    ["EDA", "KNN", "Naive Bayes", "Logistic Regression", "Random Forest Classifier"]
)

# Display the selected page
if selected_page == "EDA":
    eda_page()
elif selected_page == "KNN":
    knn_page()
elif selected_page == "Naive Bayes":
    nb_page()
elif selected_page == "Logistic Regression":
    logistic_page()
elif selected_page == "Random Forest Classifier":
    rf_page()
