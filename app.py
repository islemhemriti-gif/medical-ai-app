import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Medical AI Diagnostic System",
    page_icon="🧬",
    layout="wide"
)

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

    This app helps you analyze simple medical indicators.

    👉 Go to **Prediction** to enter your lab results  
    👉 Go to **Data Analysis** to explore dataset insights  

    ⚠️ This tool is for educational purposes only (not a medical diagnosis)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("Model based on medical dataset")

    with col2:
        st.success("Ready for use")

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":

    st.title("🧪 Patient Analysis")

    st.markdown("### Enter your lab values:")

    col1, col2, col3 = st.columns(3)

    with col1:
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)

    with col2:
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col3:
        age = st.number_input("Age", 1, 120, 30)

    st.markdown("---")

    if st.button("Analyze"):

        st.subheader("📊 Results:")

        # Glucose interpretation
        if glucose < 70:
            st.warning("⚠️ Low glucose (hypoglycemia)")
        elif glucose <= 140:
            st.success("✅ Glucose level is normal")
        else:
            st.error("⚠️ High glucose (possible diabetes risk)")

        # BMI interpretation
        if bmi < 18.5:
            st.info("BMI: Underweight")
        elif bmi < 25:
            st.success("BMI: Normal")
        elif bmi < 30:
            st.warning("BMI: Overweight")
        else:
            st.error("BMI: Obesity")

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

    st.markdown("### Preview of dataset")

    # Load dataset
    df = pd.read_csv("diabetes.csv.csv")

    st.dataframe(df.head())

    st.markdown("---")

    st.markdown("### 📈 Data Visualization")

    col1, col2 = st.columns(2)

    # Glucose chart
    with col1:
        st.subheader("Glucose Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(df["Glucose"], bins=20)
        st.pyplot(fig1)

    # BMI chart
    with col2:
        st.subheader("BMI Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["BMI"], bins=20)
        st.pyplot(fig2)

    st.markdown("---")

    # Age chart
    st.subheader("Age Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(df["Age"], bins=20)
    st.pyplot(fig3)
