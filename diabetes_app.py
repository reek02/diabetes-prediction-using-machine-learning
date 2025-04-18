import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

# Page setup
st.set_page_config(page_title="🩺 Advanced Diabetes Predictor", layout="wide", page_icon="🧬")

# Custom CSS for enhanced UI with black font in light mode
st.markdown("""
    <style>
        /* Global dark background and text */
        .stApp {
            background-color: #1e1e1e;
            color: #f0f0f0;
            padding: 20px;
            border-radius: 12px;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #4FC3F7;
            text-align: center;
        }

        /* Sidebar style */
        section[data-testid="stSidebar"] {
            background-color: #2c2c2c;
            border-right: 1px solid #444;
        }

        /* Sidebar text */
        .sidebar .sidebar-content {
            color: #f0f0f0;
        }

        /* Input fields */
        .stNumberInput > div > input {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #555;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #4FC3F7;
            color: #000000;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 8px;
            transition: 0.3s;
        }

        div.stButton > button:hover {
            background-color: #0288D1;
            color: white;
        }

        /* Download button */
        .stDownloadButton {
            background-color: #81C784 !important;
            color: #ffffff !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }

        .stDownloadButton:hover {
            background-color: #66BB6A !important;
            color: #ffffff !important;
        }


        /* Chart and canvas containers */
        .element-container:has(canvas),
        .element-container:has(svg) {
            background-color: #292929;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        }

        /* Expander */
        details {
            background-color: #2c2c2c;
            border-radius: 10px;
            padding: 10px;
            margin-top: 20px;
            box-shadow: 0 0 5px rgba(255,255,255,0.05);
        }

        /* Footer */
        footer {
            text-align: center;
            color: #999;
            font-size: 0.9rem;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown("<h1>🩺 Advanced Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("### Enter patient information in the sidebar to check their diabetes risk.")

# Sidebar for user inputs
st.sidebar.header("📋 Enter Medical Information")
pregnancies = st.sidebar.number_input("Pregnancies (0–20)", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose Level (0–200)", min_value=0, max_value=200, value=110)
blood_pressure = st.sidebar.number_input("Blood Pressure (0–200)", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness (0–100)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level (0–900)", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI (0.0–70.0)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function (0.0–3.0)", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.sidebar.number_input("Age (1–120)", min_value=1, max_value=120, value=30)

# Input ranges reference
st.sidebar.markdown("""
- **Pregnancies**: 0 to 20  
- **Glucose Level**: 0 to 200  
- **Blood Pressure**: 0 to 200  
- **Skin Thickness**: 0 to 100  
- **Insulin Level**: 0 to 900  
- **BMI**: 0.0 to 70.0  
- **Diabetes Pedigree Function**: 0.0 to 3.0  
- **Age**: 1 to 120  
""")

# Load model
model = joblib.load("D:/Projects/tryouts/diabetes_prediction/random_forest_model.joblib")

if st.sidebar.button("🔍 Predict"):
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Layout columns
    col1, col2 = st.columns([2, 1])

    # Prediction result
    with col1:
        st.subheader("🔎 Prediction Result")
        if prediction == 1:
            st.error("🚨 The person is likely **Diabetic**.")
        else:
            st.success("✅ The person is likely **Healthy**.")
        st.info(f"📊 Prediction Confidence: **{probability*100:.2f}%**")

        # Personalized recommendations
        st.subheader("💡 Personalized Health Recommendations")
        if prediction == 1:
            st.warning("⚠️ You are at risk of diabetes. Please consult a healthcare professional.")
            st.markdown("""
- **Diet**: Rich in vegetables, whole grains, lean proteins  
- **Exercise**: ≥30 min moderate activity most days  
- **Monitoring**: Check blood glucose regularly  
- **Consultation**: Schedule regular check‑ups  
""")
        else:
            st.success("🎉 You are likely healthy!")
            st.markdown("""
- **Diet**: Maintain a balanced diet  
- **Exercise**: Continue regular activity  
- **Monitoring**: Track key health metrics  
- **Consultation**: Keep up routine check‑ups  
""")

    # Probability pie chart
    with col2:
        st.subheader("📈 Probability Chart")
        fig, ax = plt.subplots()
        ax.pie([probability, 1 - probability],
               labels=["Diabetic", "Healthy"],
               autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    # PDF summary generation
    def create_pdf():
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Diabetes Prediction Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, f"""\
Pregnancies: {pregnancies}
Glucose Level: {glucose}
Blood Pressure: {blood_pressure}
Skin Thickness: {skin_thickness}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}
--------------------------
Prediction: {'Diabetic' if prediction else 'Healthy'}
Confidence: {probability*100:.2f}%
""")
        return pdf.output(dest='S').encode('latin1')

    pdf_data = create_pdf()
    st.download_button("📥 Download Summary as PDF",
                       data=pdf_data,
                       file_name="prediction_summary.pdf",
                       mime="application/pdf")

    # Feature importance plot
    st.subheader("🔍 Feature Importance")
    feature_names = [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance - Gini Importance")
    st.pyplot(fig)

# Model info expander
with st.expander("ℹ️ About the Model"):
    st.markdown("""
- **Model**: Random Forest Classifier  
- **Trained on**: Kaggle Pima Indians Diabetes Dataset  
- **Features used**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age  
- **Accuracy**: High performance with balanced metrics  
- **Interpretation**: Ensemble of decision trees identifying key patterns for reliable risk prediction.
""")

# Footer
st.markdown("---")
st.markdown("<footer>🔬 Developed with ❤️ by Reekparna Sen for health prediction using ML</footer>", unsafe_allow_html=True)
