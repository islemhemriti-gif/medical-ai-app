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

    mode = st.radio("Login System", ["Login", "Sign Up"])

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

# ---------------- NAVIGATION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- STYLE (HOME DASHBOARD LOOK) ----------------
st.markdown("""
<style>
.big-card {
    padding: 25px;
    border-radius: 20px;
    background-color: #111827;
    color: white;
    text-align: center;
    cursor: pointer;
    transition: 0.3s;
    border: 1px solid #1f2937;
}
.big-card:hover {
    transform: scale(1.05);
    border: 1px solid #38bdf8;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 AI Medical System")
st.sidebar.write(f"User: {st.session_state.user}")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
if st.sidebar.button("🧪 Prediction"):
    st.session_state.page = "Prediction"
if st.sidebar.button("📁 History"):
    st.session_state.page = "History"
if st.sidebar.button("📊 Data"):
    st.session_state.page = "Data"
if st.sidebar.button("📚 Sources"):
    st.session_state.page = "Sources"

page = st.session_state.page

# ---------------- HOME DASHBOARD ----------------
if page == "Home":

    st.markdown('<div class="title">🧬 Medical AI Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome 👋

    This is an AI-powered clinical decision support system.

    It predicts diabetes risk using medical data and machine learning.
    """)

    st.markdown("---")
    st.markdown("### 🚀 Choose a module")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧪 Prediction Engine"):
            st.session_state.page = "Prediction"

    with col2:
        if st.button("📁 Patient History"):
            st.session_state.page = "History"

    with col3:
        if st.button("📊 Data Analysis"):
            st.session_state.page = "Data"

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        if st.button("📚 Sources"):
            st.session_state.page = "Sources"

    with col5:
        st.info("AI System Active")

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🧪 Clinical Prediction Engine")

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

        add_history(st.session_state.user, glucose, bmi, age, float(proba))

    # IMAGE (still inside prediction ONLY)
    st.markdown("---")
    st.subheader("🖼️ Skin Image (Experimental)")

    img = st.file_uploader("Upload image", type=["jpg", "png"])

    if img is not None:
        image = Image.open(img)
        st.image(image, use_container_width=True)
        st.info("Future CNN model upgrade")

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
    - Standard clinical risk factors (glucose, BMI, age)
    """)
