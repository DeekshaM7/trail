# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("Finaldf-2.csv")

# Feature columns
X = df[['amt', 'lat', 'long', 'city_pop', 'gender', 'state', 'street', 'city', 'job', 'category']]
y = df['is_fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to NumPy arrays
X_train = X_train.values
y_train = y_train.values

# Create and train your machine learning models
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

def knn_page():
    st.title('K-Nearest Neighbors (KNN) Page')
    
    # Add UI elements for KNN model
    st.write('This page uses K-Nearest Neighbors to predict fraud.')
    
    # Collect input features from the user
    amt = st.slider('Transaction Amount', X['amt'].min(), X['amt'].max())
    
    # Collect and encode categorical features
    gender = st.radio('Gender', list(df['gender'].unique()))
    state = st.radio('State', list(df['state'].unique()))
    street = st.radio('Street', list(df['street'].unique()))
    city = st.radio('City', list(df['city'].unique()))
    job = st.radio('Job', list(df['job'].unique()))
    category = st.radio('Category', list(df['category'].unique()))
    
    # Encode categorical features
    le = LabelEncoder()
    gender_enc = le.fit_transform([gender])
    state_enc = le.fit_transform([state])
    street_enc = le.transform([street])
    city_enc = le.transform([city])
    job_enc = le.transform([job])
    category_enc = le.transform([category])

    # Create a feature array with the user's input
    features = np.array([[amt, gender_enc[0], state_enc[0], street_enc[0], city_enc[0], job_enc[0], category_enc[0]]])

    # Make predictions using the KNN model
    prediction = knn_model.predict(features)

    predict_button = st.button("Predict")

    if predict_button:
        if prediction == 1:
            prediction_txt = "Fraud Detected"
        else:
            prediction_txt = "No Fraud Detected"

        # Display the prediction
        st.write(f'The prediction is: {prediction_txt}')

def nb_page():
    st.title('Gaussian Naive Bayes Page')
    
    # Add UI elements for Gaussian Naive Bayes model
    st.write('This page uses Gaussian Naive Bayes to predict fraud.')
    
    # Collect input features from the user
    amt = st.slider('Transaction Amount', X['amt'].min(), X['amt'].max())
    
    # Collect and encode categorical features
    gender = st.radio('Gender', list(df['gender'].unique()))
    state = st.radio('State', list(df['state'].unique()))
    street = st.radio('Street', list(df['street'].unique()))
    city = st.radio('City', list(df['city'].unique()))
    job = st.radio('Job', list(df['job'].unique()))
    category = st.radio('Category', list(df['category'].unique()))
    
    # Encode categorical features
    le = LabelEncoder()
    gender_enc = le.fit_transform([gender])
    state_enc = le.fit_transform([state])
    street_enc = le.transform([street])
    city_enc = le.transform([city])
    job_enc = le.transform([job])
    category_enc = le.transform([category])

    # Create a feature array with the user's input
    features = np.array([[amt, gender_enc[0], state_enc[0], street_enc[0], city_enc[0], job_enc[0], category_enc[0]]])

    # Make predictions using Gaussian Naive Bayes model
    prediction = nb_model.predict(features)

    predict_button = st.button("Predict")

    if predict_button:
        if prediction == 1:
            prediction_txt = "Fraud Detected"
        else:
            prediction_txt = "No Fraud Detected"

        # Display the prediction
        st.write(f'The prediction is: {prediction_txt}')

def logistic_page():
    st.title('Logistic Regression Page')
    
    # Add UI elements for Logistic Regression model
    st.write('This page uses Logistic Regression to predict fraud.')
    
    # Collect input features from the user
    amt = st.slider('Transaction Amount', X['amt'].min(), X['amt'].max())
    
    # Collect and encode categorical features
    gender = st.radio('Gender', list(df['gender'].unique()))
    state = st.radio('State', list(df['state'].unique()))
    street = st.radio('Street', list(df['street'].unique()))
    city = st.radio('City', list(df['city'].unique()))
    job = st.radio('Job', list(df['job'].unique()))
    category = st.radio('Category', list(df['category'].unique()))
    
    # Encode categorical features
    le = LabelEncoder()
    gender_enc = le.fit_transform([gender])
    state_enc = le.fit_transform([state])
    street_enc = le.transform([street])
    city_enc = le.transform([city])
    job_enc = le.transform([job])
    category_enc = le.transform([category])

    # Create a feature array with the user's input
    features = np.array([[amt, gender_enc[0], state_enc[0], street_enc[0], city_enc[0], job_enc[0], category_enc[0]]])

    # Make predictions using Logistic Regression model
    prediction = logistic_model.predict(features)

    predict_button = st.button("Predict")

    if predict_button:
        if prediction == 1:
            prediction_txt = "Fraud Detected"
        else:
            prediction_txt = "No Fraud Detected"

        # Display the prediction
        st.write(f'The prediction is: {prediction_txt}')

def rf_page():
    st.title('Random Forest Page')
    
    # Add UI elements for Random Forest model
    st.write('This page uses Random Forest to predict fraud.')
    
    # Collect input features from the user
    amt = st.slider('Transaction Amount', X['amt'].min(), X['amt'].max())
    
    # Collect and encode categorical features
    gender = st.radio('Gender', list(df['gender'].unique()))
    state = st.radio('State', list(df['state'].unique()))
    street = st.radio('Street', list(df['street'].unique()))
    city = st.radio('City', list(df['city'].unique()))
    job = st.radio('Job', list(df['job'].unique()))
    category = st.radio('Category', list(df['category'].unique()))
    
    # Encode categorical features
    le = LabelEncoder()
    gender_enc = le.fit_transform([gender])
    state_enc = le.fit_transform([state])
    street_enc = le.transform([street])
    city_enc = le.transform([city])
    job_enc = le.transform([job])
    category_enc = le.transform([category])

    # Create a feature array with the user's input
    features = np.array([[amt, gender_enc[0], state_enc[0], street_enc[0], city_enc[0], job_enc[0], category_enc[0]]])

    # Make predictions using Random Forest model
    prediction = rf_model.predict(features)

    predict_button = st.button("Predict")

    if predict_button:
        if prediction == 1:
            prediction_txt = "Fraud Detected"
        else:
            prediction_txt = "No Fraud Detected"

        # Display the prediction
        st.write(f'The prediction is: {prediction_txt}')

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox("Choose a model:", ["KNN", "Logistic Regression", "Naive Bayes", "Random Forest"])

    if page == "KNN":
        knn_page()
    elif page == "Logistic Regression":
        logistic_page()
    elif page == "Naive Bayes":
        nb_page()
    elif page == "Random Forest":
        rf_page()

if _name_ == '_main_':
    main()
