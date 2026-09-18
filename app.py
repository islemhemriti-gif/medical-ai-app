import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

st.set_page_config(
    page_title="Medical AI Diagnostic System",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Medical AI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Data Analysis"]
)

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("🧬 Medical AI Diagnostic System")

    st.markdown("""
    ### Welcome 👋

    This app helps you analyze medical data using Artificial Intelligence.

    👉 Go to **Prediction** to enter your lab results  
    👉 Go to **Data Analysis** to explore dataset insights  

    ⚠️ This tool is for educational purposes only (not a medical diagnosis)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("AI model trained on medical dataset")

    with col2:
        st.success("Ready for prediction")

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":

    st.title("🧪 AI Diabetes Prediction")

    st.markdown("### Enter patient medical values:")

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
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    st.markdown("---")

    if st.button("Predict Risk"):

        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]

        st.subheader("📊 Results:")

        if prediction[0] == 1:
            st.error(f"⚠️ High Diabetes Risk ({round(probability*100,2)}%)")
        else:
            st.success(f"✅ Low Diabetes Risk ({round(probability*100,2)}%)")

    st.markdown("---")

    # Image upload
    st.markdown("### 🖼️ Optional: Upload skin image")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.info("AI skin analysis coming soon...")

# ---------------- DATA ANALYSIS PAGE ----------------
elif page == "Data Analysis":
    st.title("📊 Dataset Overview")

    try:
        df = pd.read_csv("diabetes.csv")

        st.markdown("### Dataset Preview")
        st.dataframe(df.head())

        st.markdown("---")
        st.markdown("### 📈 Data Visualization")

        col1, col2 = st.columns(2)

        # Glucose
        with col1:
            st.subheader("Glucose Distribution")
            fig1, ax1 = plt.subplots()
            ax1.hist(df["Glucose"], bins=20)
            st.pyplot(fig1)

        # BMI
        with col2:
            st.subheader("BMI Distribution")
            fig2, ax2 = plt.subplots()
            ax2.hist(df["BMI"], bins=20)
            st.pyplot(fig2)

        st.markdown("---")

        # Age
        st.subheader("Age Distribution")
        fig3, ax3 = plt.subplots()
        ax3.hist(df["Age"], bins=20)
        st.pyplot(fig3)

    except FileNotFoundError:
        st.error("❌ Dataset file not found. Make sure 'diabetes.csv' is in your repo.")
