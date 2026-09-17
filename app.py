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

    This application uses Machine Learning to analyze medical data.

    ---
    **Features:**
    - 📊 Data visualization  
    - 🤖 AI-based prediction  
    - 📈 Interactive analysis  

    ⚠️ Educational purposes only
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("Model trained on medical dataset")

    with col2:
        st.success("Ready for prediction")

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":

    st.title("Patient Analysis")

    glucose = st.number_input("Glucose", 0, 300, 120)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Analyze"):
        if glucose > 140:
            st.error("⚠️ High diabetes risk")
        else:
            st.success("✅ Low diabetes risk")

# ---------------- DATA PAGE ----------------
elif page == "Data Analysis":
    st.title("Dataset Overview")
    st.write("Add dataset visualization here later")
if st.button("Analyze"):

    if glucose < 70:
        st.warning("⚠️ Low glucose (hypoglycemia)")
    elif glucose <= 140:
        st.success("✅ Glucose level is normal")
    else:
        st.error("⚠️ High glucose (possible diabetes risk)")

    # BMI explanation
    if bmi < 18.5:
        st.info("BMI: Underweight")
    elif bmi < 25:
        st.success("BMI: Normal")
    elif bmi < 30:
        st.warning("BMI: Overweight")
    else:
        st.error("BMI: Obesity")
uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file)
    st.write("AI analysis coming soon...")
