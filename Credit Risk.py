import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
def load_data():
    df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\loan_data.csv")
    return df

# Preprocess Data
def preprocess_data(df):
    df = df.dropna()
    label_enc = LabelEncoder()
    df["Education"] = label_enc.fit_transform(df["Education"])
    df["Property_Area"] = label_enc.fit_transform(df["Property_Area"])
    return df

# Train Model
def train_model(df):
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", acc)
    
    joblib.dump(model, "credit_risk_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

# Load Trained Model
def load_model():
    model = joblib.load("credit_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# Train and Save Model
df = load_data()
df = preprocess_data(df)
train_model(df)

# Streamlit UI
st.title("Credit Risk Assessment")

st.sidebar.header("Enter Loan Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=2000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=50000)
credit_history = st.sidebar.selectbox("Credit History", [0, 1])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.sidebar.button("Predict Loan Risk"):
    model, scaler = load_model()
    
    input_data = np.array([[income, loan_amount, credit_history, int(education == "Graduate"), int(property_area == "Urban")]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("Low-Risk: Loan Approved")
    else:
        st.error("High-Risk: Loan Rejected")
