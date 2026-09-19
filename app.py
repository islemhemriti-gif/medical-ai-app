import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

from db import create_tables, add_user, login_user, add_history, get_history

# ---------------- INIT ----------------
create_tables()

st.set_page_config(
    page_title="Medical AI Platform",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- AUTH ----------------
if st.session_state.user is None:

    st.title("🧬 Medical AI Platform")

    mode = st.radio("Choose mode", ["Login", "Sign Up"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Sign Up":
        if st.button("Create Account"):
            if add_user(username, password):
                st.success("Account created")
            else:
                st.error("User already exists")

    else:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.user = username
                st.success("Welcome " + username)
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 AI Medical System")
st.sidebar.write(f"User: {st.session_state.user}")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "History", "Data", "Sources"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("AI Clinical Decision Support System")

    st.markdown("""
    ### Welcome 👋
    Predict diabetes risk using machine learning.

    ⚠️ Educational use only
    """)

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🧪 Clinical Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)

    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col3:
        dpf = st.number_input("Diabetes Pedigree", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    # ---------------- PREDICTION BUTTON ----------------
    if st.button("Run AI Analysis"):

        data = np.array([[pregnancies, glucose, blood_pressure,
                          skin, insulin, bmi, dpf, age]])

        scaled = scaler.transform(data)

        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk %", f"{round(proba*100,2)}%")
        col2.metric("Glucose", glucose)
        col3.metric("BMI", bmi)

        if pred == 1:
            st.error("⚠️ High Risk")
        else:
            st.success("Low Risk")

        # save history
        add_history(st.session_state.user, glucose, bmi, age, float(proba))

    # ---------------- IMAGE UPLOAD (ONLY HERE) ----------------
    st.markdown("---")
    st.subheader("🖼️ Skin Image Analysis (Experimental)")

    uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.info("AI skin analysis coming soon (CNN model upgrade)")

# ---------------- HISTORY ----------------
elif page == "History":

    st.title("Patient History")

    data = get_history(st.session_state.user)

    if data:
        df = pd.DataFrame(data, columns=["Glucose", "BMI", "Age", "Risk"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history found")

# ---------------- DATA ----------------
elif page == "Data":

    st.title("Dataset Overview")

    df = pd.read_csv("diabetes.csv")

    st.bar_chart(df["Glucose"])
    st.bar_chart(df["BMI"])

    st.dataframe(df.corr())

# ---------------- SOURCES ----------------
elif page == "Sources":

    st.title("Medical Sources")

    st.markdown("""
    - PIMA Diabetes Dataset (UCI)
    - WHO Clinical Guidelines
    - ML-based risk factors (glucose, BMI, age)
    """)
