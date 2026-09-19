import streamlit as st
import pandas as pd
import numpy as np
import joblib

from db import create_tables, add_user, login_user, add_history, get_history

# ---------------- INIT ----------------
create_tables()

st.set_page_config(
    page_title="Medical AI Platform",
    page_icon="🧬",
    layout="wide"
)

# ---------------- STYLE (PREMIUM UI) ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}

.main {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stMetric {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}

.css-1d391kg {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- AUTH ----------------
if st.session_state.user is None:

    st.title("🧬 Medical AI Platform")

    st.markdown("### AI Clinical Decision Support System")

    mode = st.radio("Choose mode", ["Login", "Sign Up"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Sign Up":
        if st.button("Create Account"):
            if add_user(username, password):
                st.success("Account created successfully")
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
st.sidebar.title("🧬 Medical AI System")
st.sidebar.write(f"User: **{st.session_state.user}**")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧪 Prediction", "📁 History", "📊 Data", "📚 Sources"]
)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("AI Clinical Decision Support Platform")

    st.markdown("""
    ### Welcome 👋

    This system predicts diabetes risk using machine learning trained on clinical data.

    ---
    ### Features:
    - AI Prediction Engine
    - User-based history tracking
    - Clinical report generation
    - Data visualization dashboard

    ⚠️ Educational use only
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Dataset", "PIMA Indians")
    col3.metric("System", "Active")

# ---------------- PREDICTION ----------------
elif page == "🧪 Prediction":

    st.title("Clinical Prediction Engine")

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

        st.markdown("### Results Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk %", f"{round(proba*100,2)}%")
        c2.metric("Glucose", glucose)
        c3.metric("BMI", bmi)

        if pred == 1:
            st.error("⚠️ High Diabetes Risk")
        else:
            st.success("Low Risk")

        # ---------------- SAVE HISTORY ----------------
        add_history(
            st.session_state.user,
            glucose,
            bmi,
            age,
            float(proba)
        )

# ---------------- HISTORY ----------------
elif page == "📁 History":

    st.title("Patient History")

    data = get_history(st.session_state.user)

    if data:
        df = pd.DataFrame(data, columns=["Glucose", "BMI", "Age", "Risk"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history found")

# ---------------- DATA ----------------
elif page == "📊 Data":

    st.title("Dataset Analysis")

    df = pd.read_csv("diabetes.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df["Glucose"])

    with col2:
        st.bar_chart(df["BMI"])

    st.subheader("Correlation Matrix")
    st.dataframe(df.corr())

# ---------------- SOURCES ----------------
elif page == "📚 Sources":

    st.title("Medical & Data Sources")

    st.markdown("""
    - PIMA Indians Diabetes Dataset (UCI)
    - WHO Diabetes Guidelines
    - Clinical Risk Factors: glucose, BMI, age, insulin

    ⚠️ This is an educational simulation, not a medical device.
    """)

if img:
    image = Image.open(img)
    st.image(image, use_container_width=True)
    st.info("Future upgrade: CNN-based medical imaging model")
