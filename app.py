import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph

st.set_page_config(
    page_title="Medical AI Platform",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Medical AI Platform")
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "Data Analysis"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🧬 Medical AI Platform")

    st.markdown("""
    ### AI-Powered Clinical Decision Support

    This platform analyzes medical data using machine learning to assist in early diabetes risk detection.

    **Modules:**
    - 🧪 Prediction Engine  
    - 📊 Data Analysis Dashboard  
    - 📄 Report Generation  

    ⚠️ Educational use only
    """)

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🧪 Clinical Prediction Engine")

    st.markdown("### Enter patient clinical parameters")

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

    if st.button("Run AI Analysis"):

        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        scaled = scaler.transform(input_data)

        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        st.subheader("📊 Clinical Results")

        # Metrics (PRO UI)
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk (%)", f"{round(proba*100,2)}%")
        m2.metric("Glucose", glucose)
        m3.metric("BMI", bmi)

        if prediction == 1:
            st.error("⚠️ High Diabetes Risk")
        else:
            st.success("✅ Low Diabetes Risk")

        # ---------------- EXPLANATION ----------------
        st.markdown("### 🧠 Clinical Interpretation")

        report_points = []

        if glucose > 140:
            report_points.append("Elevated glucose level indicates impaired glycemic control.")
        if bmi > 30:
            report_points.append("BMI suggests obesity, a major diabetes risk factor.")
        if age > 45:
            report_points.append("Age increases metabolic risk.")
        if dpf > 0.8:
            report_points.append("Family history contributes to genetic predisposition.")

        if report_points:
            for p in report_points:
                st.write("•", p)
        else:
            st.write("No strong clinical risk factors identified.")

        # ---------------- PDF ----------------
        report_text = f"""
        AI CLINICAL REPORT

        Risk: {round(proba*100,2)}%
        Glucose: {glucose}
        BMI: {bmi}
        Age: {age}
        """

        pdf = SimpleDocTemplate("report.pdf", pagesize=letter)
        content = [Paragraph(report_text)]
        pdf.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("📄 Download Clinical Report", f, "report.pdf")

    # Image
    st.markdown("---")
    st.markdown("### 🖼️ Dermatology (Experimental)")

    uploaded = st.file_uploader("Upload skin image", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
        st.info("AI dermatology module coming soon...")

# ---------------- DATA ----------------
elif page == "Data Analysis":
    st.title("📊 Data Analysis Dashboard")

    df = pd.read_csv("diabetes.csv")

    st.dataframe(df.head())

    st.markdown("### 📈 Feature Distributions")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.hist(df["Glucose"], bins=20)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.hist(df["BMI"], bins=20)
        st.pyplot(fig2)

    st.markdown("### 🔬 Correlation Heatmap")

    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)
