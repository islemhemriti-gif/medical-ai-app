import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

from db import create_tables, add_user, login_user, add_history, get_history

# ---------------- INIT ----------------
create_tables()

st.set_page_config(
    page_title="MedAI - Clinical Intelligence Platform",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- AUTH ----------------
if st.session_state.user is None:

    st.title("🧬 MedAI Platform")

    mode = st.radio("Access System", ["Login", "Sign Up"])

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

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top,#0f172a,#020617);
    color: white;
}

.title {
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(90deg,#38bdf8,#60a5fa,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(145deg,#1e293b,#0f172a);
    border: 1px solid #334155;
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#38bdf8);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 MedAI System")
st.sidebar.write(f"User: {st.session_state.user}")

pages = ["Home", "Prediction", "History", "Data", "Sources"]
choice = st.sidebar.radio("Navigation", pages)

st.session_state.page = choice
page = choice

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.session_state.page = "Home"
    st.rerun()

# ---------------- HOME ----------------
if page == "Home":

    st.markdown('<div class="title">MedAI Clinical Intelligence</div>', unsafe_allow_html=True)

    st.markdown("AI-powered clinical decision support system")

    st.markdown("---")

    st.info("🧠 Diabetes ML System + 📊 Medical Analytics (CNN coming soon)")

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🧪 Diabetes Prediction Engine")

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

        st.success(f"Risk Score: {round(proba*100,2)}%")

        if pred == 1:
            st.error("⚠️ High Risk")
        else:
            st.success("Low Risk")

        add_history(st.session_state.user, glucose, bmi, age, float(proba))

    st.markdown("---")

    # ---------------- SKIN AI (DISABLED SAFE VERSION) ----------------
    st.subheader("🖼 Skin AI Detection (Coming Soon 🚀)")

    st.info("""
    This feature will include:
    - CNN Deep Learning model
    - Skin disease classification
    - Real medical imaging AI

    ⚠️ Currently disabled for cloud deployment stability
    """)

    img = st.file_uploader("Upload image (preview only)", type=["jpg", "png"])

    if img:
        image = Image.open(img)
        st.image(image, use_container_width=True)
        st.warning("AI model not deployed in cloud version")

# ---------------- HISTORY ----------------
elif page == "History":

    st.title("📁 Patient History")

    data = get_history(st.session_state.user)

    if data:
        df = pd.DataFrame(data, columns=["Glucose", "BMI", "Age", "Risk"])
        st.dataframe(df)
    else:
        st.info("No history")

# ---------------- DATA ----------------
elif page == "Data":

    st.title("📊 Dataset Overview")

    df = pd.read_csv("diabetes.csv")

    st.bar_chart(df["Glucose"])
    st.bar_chart(df["BMI"])

# ---------------- SOURCES ----------------
elif page == "Sources":

    st.title("📚 Medical References")

    st.markdown("""
    - PIMA Diabetes Dataset (UCI)
    - WHO Clinical Guidelines
    - Machine Learning for Medical Prediction
    """)
