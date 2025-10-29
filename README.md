# medical-diagnosis-system
medical diagnosis system using neural networks

# 🧠 Medical Diagnosis System using Neural Networks  

## 🩺 Overview  
This project is an **AI-powered Medical Diagnosis System** built using **Neural Networks**.  
It predicts the likelihood of major diseases based on user input and medical parameters.  
The goal is to assist doctors and patients in early disease detection and risk assessment.  

---

## ⚙️ Technologies Used  
- **Python** 🐍  
- **TensorFlow / Keras** – for building and training neural networks  
- **scikit-learn** – for preprocessing and evaluation  
- **Pandas, NumPy** – for data handling  
- **Streamlit** – for creating an interactive web UI  
- **Matplotlib / Seaborn** – for EDA and visualization  

---

## 🧩 Diseases Covered  
✅ Brain Tumor Detection  
✅ Migraine Prediction  
✅ Kidney Disease Detection  
✅ Liver Disease Prediction  
✅ Lung Disease Detection  
✅ Heart Disease Prediction  

*(All models integrated into a single unified Streamlit UI.)*

---

## 🧠 Model Details  
Each disease prediction uses an individual neural network trained on its corresponding dataset.  
The system includes preprocessing pipelines for each dataset (normalization, encoding, scaling, etc.)  
and produces a prediction probability (0 → Healthy, 1 → Diseased).  

---

## 💻 Features  
- 🔹 Easy-to-use Streamlit web interface  
- 🔹 Predict multiple diseases from a single dashboard  
- 🔹 Generate downloadable medical reports (PDF format)  
- 🔹 Attractive UI with professional design and medical theme  
- 🔹 Modular code structure for future expansion  

---
## 🚀 How to Run  

### 1️⃣ Clone the repository


##    git clone git@github.com:vvishwarajput/medical-diagnosis-system.git

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows


3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
streamlit run app.py

dataset link - 1) heart disease - https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
               2) lung disease - https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types
               3) liver disease - https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset

