import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Medical AI Platform", layout="wide")

# ---------------- FILES ----------------
USERS_FILE = "users.csv"
HISTORY_FILE = "history.csv"

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HELPERS ----------------
def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False
    df = pd.concat([df, pd.DataFrame([{"username": username, "password": password}])])
    df.to_csv(USERS_FILE, index=False)
    return True

def save_history(row):
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([old, df])
    df.to_csv(HISTORY_FILE, index=False)

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

# ---------------- AUTH SYSTEM ----------------
if not st.session_state.logged_in:

    st.title("🔐 Medical AI Platform")

    auth_mode = st.selectbox("Choose option", ["Login", "Sign Up"])

    if auth_mode == "Sign Up":
        st.subheader("Create account")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Create Account"):
            if save_user(new_user, new_pass):
                st.success("Account created successfully")
            else:
                st.error("Username already exists")

    else:
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            users = load_users()
            if ((users["username"] == username) &
                (users["password"] == password)).any():

                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome {username}")
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Medical AI Platform")
st.sidebar.write(f"Logged in as: **{st.session_state.user}**")

page = st.sidebar.radio("Navigation", ["Home", "Prediction", "History", "Data", "Sources"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🧬 Medical AI Clinical Decision System")

    st.markdown("""
    ### AI-powered diabetes risk prediction platform

    This system uses machine learning trained on medical datasets
    to estimate diabetes risk and generate clinical reports.

    ⚠️ Educational purpose only
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Dataset", "PIMA Indians")
    col3.metric("Type", "Clinical AI")

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

        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin, insulin, bmi, dpf, age]])

        scaled = scaler.transform(input_data)

        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        st.subheader("📊 AI Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk %", f"{round(proba*100,2)}%")
        col2.metric("Glucose", glucose)
        col3.metric("BMI", bmi)

        if pred == 1:
            st.error("⚠️ High Diabetes Risk")
        else:
            st.success("✅ Low Risk")

        # ---------------- SAVE HISTORY ----------------
        save_history({
            "user": st.session_state.user,
            "glucose": glucose,
            "bmi": bmi,
            "age": age,
            "risk": round(proba*100,2)
        })

        # ---------------- REPORT ----------------
        report_text = f"""
        MEDICAL AI REPORT

        Patient: {st.session_state.user}

        Glucose: {glucose}
        BMI: {bmi}
        Age: {age}

        Risk: {round(proba*100,2)}%

        Interpretation:
        {"High risk detected" if pred==1 else "Low risk detected"}

        Source:
        PIMA Indians Diabetes Dataset (UCI Machine Learning Repository)
        """

        pdf = SimpleDocTemplate("report.pdf", pagesize=letter)
        content = []

        for line in report_text.split("\n"):
            content.append(Paragraph(line))
            content.append(Spacer(1, 6))

        pdf.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("📄 Download Report", f, "report.pdf")

# ---------------- HISTORY ----------------
elif page == "History":
    st.title("📁 Patient History")

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df)
    else:
        st.info("No history yet")

# ---------------- DATA ----------------
elif page == "Data":
    st.title("📊 Dataset Analysis")

    df = pd.read_csv("diabetes.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df["Glucose"])

    with col2:
        st.bar_chart(df["BMI"])

    st.subheader("Correlation Heatmap")
    st.dataframe(df.corr())

# ---------------- SOURCES ----------------
elif page == "Sources":
    st.title("📚 Data & Medical Sources")

    st.markdown("""
    **Dataset used:**
    - PIMA Indians Diabetes Dataset (UCI Machine Learning Repository)

    **Medical reference basis:**
    - WHO guidelines for diabetes risk factors
    - Clinical indicators: glucose, BMI, age, insulin levels

    **Disclaimer:**
    This tool is not a medical diagnostic device.
    It is for educational and research purposes only.
    """)

# ---------------- IMAGE ----------------
st.markdown("---")
st.markdown("### 🖼️ Experimental AI Imaging Module")

img = st.file_uploader("Upload medical image", type=["jpg", "png"])

if img:
    image = Image.open(img)
    st.image(image, use_container_width=True)
    st.info("Future upgrade: CNN-based medical imaging model")
