import streamlit as st

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

    # Image upload (ONLY HERE)
    st.markdown("### 🖼️ Optional: if you have a rash Upload skin image")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
        st.info("AI skin analysis coming soon...")
from PIL import Image

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.info("AI skin analysis coming soon...")
# ---------------- DATA PAGE ----------------
elif page == "Data Analysis":
    st.title("📊 Dataset Overview")

    st.write("This section will display dataset insights and charts.")
