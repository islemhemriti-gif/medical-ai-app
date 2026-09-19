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

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- AUTH ----------------
if st.session_state.user is None:

    st.markdown("""
    <style>
    .auth-box {
        padding: 40px;
        border-radius: 20px;
        background: linear-gradient(135deg,#0f172a,#1e293b);
        text-align: center;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-box'><h1>🧬 MedAI Platform</h1><p>Clinical Intelligence System</p></div>", unsafe_allow_html=True)

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

# ---------------- STYLE (LANDING UI) ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top,#0f172a,#020617);
    color: white;
}

/* CARD STYLE */
.card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(145deg,#1e293b,#0f172a);
    border: 1px solid #334155;
    transition: 0.3s;
    cursor: pointer;
    text-align: center;
}
.card:hover {
    transform: translateY(-8px);
    border: 1px solid #38bdf8;
    box-shadow: 0 10px 30px rgba(56,189,248,0.2);
}

/* TITLE */
.title {
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg,#38bdf8,#60a5fa,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtitle {
    color: #94a3b8;
    font-size: 18px;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg,#2563eb,#38bdf8);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 MedAI System")
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

# ---------------- HOME (LANDING PAGE) ----------------
if page == "Home":

    st.markdown('<div class="title">MedAI Clinical Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="subtitle">AI-powered diabetes risk analysis platform</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🚀 Explore the system")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>🧪<h3>Prediction Engine</h3><p>Analyze medical inputs with AI</p></div>", unsafe_allow_html=True)
        if st.button("Open Prediction"):
            st.session_state.page = "Prediction"

    with col2:
        st.markdown("<div class='card'>📁<h3>Patient History</h3><p>Track past predictions</p></div>", unsafe_allow_html=True)
        if st.button("Open History"):
            st.session_state.page = "History"

    with col3:
        st.markdown("<div class='card'>📊<h3>Data Insights</h3><p>Visualize medical dataset</p></div>", unsafe_allow_html=True)
        if st.button("Open Data"):
            st.session_state.page = "Data"

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("<div class='card'>📚<h3>Sources</h3><p>Medical references & dataset</p></div>", unsafe_allow_html=True)
        if st.button("Open Sources"):
            st.session_state.page = "Sources"

    with col5:
        st.markdown("<div class='card'>🧬<h3>Status</h3><p>System Active</p></div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🧪 AI Clinical Prediction Engine")

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
            st.error("⚠️ High Risk Detected")
        else:
            st.success("Low Risk")

        add_history(st.session_state.user, glucose, bmi, age, float(proba))

    st.markdown("---")

    st.subheader("🖼️ Optional Skin Image (Future AI Module)")
    img = st.file_uploader("Upload image", type=["jpg", "png"])

    if img:
        image = Image.open(img)
        st.image(image, use_container_width=True)
        st.info("Future upgrade: CNN medical imaging model")

# ---------------- HISTORY ----------------
elif page == "History":

    st.title("📁 Patient History")

    data = get_history(st.session_state.user)

    if data:
        df = pd.DataFrame(data, columns=["Glucose", "BMI", "Age", "Risk"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history available")

# ---------------- DATA ----------------
elif page == "Data":

    st.title("📊 Dataset Overview")

    df = pd.read_csv("diabetes.csv")

    st.bar_chart(df["Glucose"])
    st.bar_chart(df["BMI"])

    st.dataframe(df.corr())

# ---------------- SOURCES ----------------
elif page == "Sources":

    st.title("📚 Medical References")

    st.markdown("""
    - PIMA Indians Diabetes Dataset (UCI)
    - WHO Clinical Guidelines
    - Clinical risk factors: glucose, BMI, age
    """)
