import streamlit as st
import numpy as np
from fpdf import FPDF
from keras.preprocessing import image
import datetime
import joblib
import os
import cv2
from keras.models import load_model
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import io

# Set page configuration
st.set_page_config(page_title="Medical Diagnosis System", layout="wide")


# Function to preprocess the image


# Background setup
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

image_path = "m.jpg"

if os.path.exists(image_path):
    encoded_image = get_image_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .big-label {{
            font-size: 18px !important;
            font-weight: bold !important;
            color: white;
        }}
        .blue-header > h2 {{
            color: #b3e5fc;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.6); 
            padding: 20px;
            border-radius: 12px;
        }}
        /* Hide the Streamlit top bar */
        header {{
            visibility: hidden;
        }}
        /* Main heading styling - Left aligned, white color, shadow effects */
        .main-heading {{
            text-align: left;
            font-size: 60px;
            font-weight: bold;
            color: white;
            margin-top: 30px;
            text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.7), 0 0 25px rgba(0, 0, 0, 0.6), 0 0 5px rgba(0, 0, 0, 0.6);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Background image not found!")

# Main heading at the top - Left aligned, with white color and shadow effect
st.markdown('<div class="main-heading">Medical Diagnosis System</div>', unsafe_allow_html=True)

# Load models
heart_model = load_model("heart_disease/heart_disease_model.h5")
liver_model = joblib.load("liver_disease/liver_disease_model.pkl")
liver_scaler = joblib.load("liver_disease/scaler.pkl")
heart_scaler = joblib.load("heart_disease/heart_scaler.pkl")  # Load heart disease scaler
lung_model = load_model("lung_disease/lung_disease_model.h5")  # Load the lung disease model
lung_scaler = joblib.load("lung_disease/scaler.pkl")  # Load the scaler for lung disease




def generate_report(disease_name, input_data_dict, prediction_result):
    # Mapping of variable names to proper names
    param_names = {
        "age": "Age",
        "sex": "Sex",
        "cp": "Chest Pain Type",
        "fbs": "Fasting Blood Sugar > 120 mg/dl",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Serum Cholesterol (mg/dl)",
        "restecg": "Resting ECG Results",
        "thalach": "Max Heart Rate Achieved",
        "exang": "Exercise Induced Angina",
        "oldpeak": "Oldpeak (ST depression)",
        "slope": "Slope of Peak Exercise ST Segment",
        "tot_bilirubin": "Total Bilirubin",
        "direct_bilirubin": "Direct Bilirubin",
        "tot_proteins": "Total Proteins",
        "albumin": "Albumin",
        "ag_ratio": "Albumin/Globulin Ratio",
        "sgpt": "SGPT (ALT)",
        "sgot": "SGOT (AST)",
        "alkphos": "Alkaline Phosphatase"
    }

    # Mapping for categorical values
    value_mappings = {
        "sex": {0: "Female", 1: "Male"},
        "fbs": {0: "No", 1: "Yes"},
        "exang": {0: "No", 1: "Yes"},
        "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
        "cp": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"},
        "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        # For liver disease
        "gender": {0: "Female", 1: "Male"}
    }

    pdf = FPDF()
    pdf.add_page()

    # Set Title
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, f"Medical Diagnosis Report - {disease_name}", ln=True, align='C')

    # Timestamp
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')

    pdf.ln(10)

    # Patient Data Table
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Input Parameters", ln=True)
    
    pdf.set_fill_color(230, 230, 250)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(90, 10, "Parameter", 1, 0, 'C', fill=True)
    pdf.cell(90, 10, "Value", 1, 1, 'C', fill=True)

    pdf.set_font("Arial", '', 11)
    for key, value in input_data_dict.items():
        # Replace snake_case with human-readable names
        param_name = param_names.get(key, key).capitalize()
        
        # Map categorical values to readable values
        if key in value_mappings:
            value = value_mappings[key].get(value, value)  # Use mapped value if exists
        
        pdf.cell(90, 10, param_name, 1)
        pdf.cell(90, 10, str(value), 1, 1)

    # Prediction Result
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)

    # Set color based on result
    if "not" in prediction_result.lower():
        pdf.set_text_color(34, 139, 34)  # Green
    else:
        pdf.set_text_color(178, 34, 34)  # Red

    pdf.cell(0, 10, f"Prediction: {prediction_result}", ln=True)

    # Save the file
    filename = f"{disease_name.replace(' ', '_')}_report.pdf"
    pdf.output(filename)
    return filename






# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3771/3771518.png", width=100)
st.sidebar.title("🧪 Choose Diagnosis")
selected_disease = st.sidebar.selectbox("Select a disease", ["Heart Disease", "Liver Disease", "Lung Disease"])


# ---------------- HEART DISEASE ----------------
if selected_disease == "Heart Disease":
    st.markdown('<div class="blue-header"><h2>💓 Heart Disease Diagnosis</h2></div>', unsafe_allow_html=True)

    # Inputs for prediction
    age = st.slider("Age (in years)", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    if st.button("🔍 Predict Heart Disease"):
        # Convert inputs into appropriate values
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs == "Yes" else 0
        exang_val = 1 if exang == "Yes" else 0
        restecg_val = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[restecg]
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        slope_val = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]

        # Create input array with all 11 features
        input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                                restecg_val, thalach, exang_val, oldpeak, slope_val]])

        # Scale the input data using the scaler
        scaled_input_data = heart_scaler.transform(input_data)

        # Make the prediction
        prediction = heart_model.predict(scaled_input_data)

        # Get predicted probability
        probability = prediction[0][0]

        # Display the probability to check how confident the model is
        st.write(f"Prediction Probability: {probability:.4f}")

        # Use a more conservative threshold for prediction
        if probability > 0.5:
            result = "Patient has heart disease"
        else:
            result = "Patient does not have heart disease"

        st.success(f"**Prediction Result:** {result}")

        heart_feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']
        heart_input_dict = {name: value for name, value in zip(heart_feature_names, input_data[0])}
        pdf_file = generate_report("Heart Disease", heart_input_dict, result)

        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf_file)


# ---------------- LIVER DISEASE ----------------
elif selected_disease == "Liver Disease":
    st.markdown('<div class="blue-header"><h2>🧬 Liver Disease Diagnosis</h2></div>', unsafe_allow_html=True)

    # Inputs for prediction
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tot_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    tot_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    ag_ratio = st.number_input("Albumin/Globulin Ratio")
    sgpt = st.number_input("SGPT (ALT)")
    sgot = st.number_input("SGOT (AST)")
    alkphos = st.number_input("Alkaline Phosphotase")

    if st.button("🔬 Predict Liver Disease"):
        gender_val = 1 if gender == "Male" else 0
        input_data = np.array([[age, gender_val, tot_bilirubin, direct_bilirubin,tot_proteins,albumin,ag_ratio
                                ,sgpt, sgot,alkphos]])

        # Scale input data for liver disease prediction
        scaled_input = liver_scaler.transform(input_data)

        # Make the prediction
        prediction = liver_model.predict(scaled_input)

        # Result based on prediction
        if prediction[0] == 1:
            result = "Patient has liver disease"
        else:
            result = "Patient does not have liver disease"

        st.success(f"**Prediction Result:** {result}")

        liver_feature_names = ['age', 'gender', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']
        liver_input_dict = {name: value for name, value in zip(liver_feature_names, input_data[0])}
        pdf_file = generate_report("Liver Disease", liver_input_dict, result)

        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf_file)


# ---------------- LUNG DISEASE ----------------

# When the user selects Lung Disease
elif selected_disease == "Lung Disease":
   
    def preprocess_lung_image(uploaded_file, scaler):
        try:
            # Read and decode image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            img_flattened = img.reshape(1, -1)
            img_scaled = scaler.transform(img_flattened)
            img_scaled = img_scaled.reshape(-1, 128, 128, 3)
            return img_scaled
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return None
    

    def predict_lung_disease(img_array,model):
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)

        label_map = {
            0: 'Bacterial Pneumonia',
            1: 'Corona Virus',
            2: 'Normal'
        }

        if confidence >= 0.6:
            return f"✅ Prediction: **{label_map[predicted_class]}**\n🧠 Confidence: {confidence * 100:.2f}%"
        else:
            return "⚠️ The model is not confident in the prediction. Please upload a clearer image or try again."




    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
        img_array = preprocess_lung_image(uploaded_file, lung_scaler)
        if img_array is not None:
            result = predict_lung_disease(img_array, lung_model)
            st.markdown(result)

