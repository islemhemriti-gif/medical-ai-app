import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph

st.set_page_config(page_title="Medical AI Platform", layout="wide")

# ---------------- DARK MODE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    body {background-color: #0e1117; color: white;}
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Medical AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Logged in successfully")
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Medical AI Platform")
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "History", "Data Analysis"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🧬 Medical AI Platform")

    st.markdown("""
    ### AI Clinical Decision Support System

    This system predicts diabetes risk using machine learning.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", "~80%")
    col3.metric("Status", "Active")

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🧪 AI Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)

    with col2:
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col3:
        dpf = st.number_input("DPF", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("Run AI Analysis"):

        data = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])

        scaled = scaler.transform(data)

        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk %", f"{round(proba*100,2)}%")
        col2.metric("Glucose", glucose)
        col3.metric("BMI", bmi)

        if pred == 1:
            st.error("High Risk")
        else:
            st.success("Low Risk")

        # Save history
        new_data = pd.DataFrame([{
            "Glucose": glucose,
            "BMI": bmi,
            "Age": age,
            "Risk": proba
        }])

        if os.path.exists("history.csv"):
            old = pd.read_csv("history.csv")
            new_data = pd.concat([old, new_data])

        new_data.to_csv("history.csv", index=False)

        # PDF
        pdf = SimpleDocTemplate("report.pdf", pagesize=letter)
        content = [Paragraph(f"Risk: {round(proba*100,2)}%")]
        pdf.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("Download Report", f, "report.pdf")

# ---------------- HISTORY ----------------
elif page == "History":
    st.title("📁 Patient History")

    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df)
    else:
        st.info("No history yet")

# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    df = pd.read_csv("diabetes.csv")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["Glucose"])
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.hist(df["BMI"])
        st.pyplot(fig2)

    st.subheader("Correlation Heatmap")

    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax3)
    st.pyplot(fig3)
