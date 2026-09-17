import streamlit as st

st.title("🧬 AI Medical Lab System")

st.write("Enter patient lab values")

glucose = st.number_input("Glucose", 0, 300, 120)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
age = st.number_input("Age", 1, 120, 30)

if st.button("Analyze"):
    if glucose > 140:
        st.error("⚠️ High diabetes risk")
    else:
        st.success("✅ Low diabetes risk")
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Medical AI Diagnostic System",
    page_icon="🧬",
    layout="wide"
)
