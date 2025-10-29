import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('liver_disease_model.pkl')  # Ensure the path is correct
scaler = joblib.load('scaler.pkl')  # Ensure the path is correct

# Function to predict liver disease
def predict_liver_disease(input_data):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data], columns=[
        'age', 'gender', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
        'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos'
    ])
    
    # Handle categorical data (Gender: Female = 1, Male = 0)
    input_df['gender'] = input_df['gender'].map({'Female': 1, 'Male': 0})

    # Scale the data using the same scaler used during training
    scaled_input = scaler.transform(input_df)

    # Make prediction using the trained model
    prediction = model.predict(scaled_input)

    # Return the result
    if prediction == 1:
        return "The patient has liver disease."
    else:
        return "The patient does not have liver disease."

# Function to interact with the user and gather input
def get_input_from_user():
    print("Please enter the patient's details:")

    # Gather user input for each feature
    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    tot_bilirubin = float(input("Total Bilirubin: "))
    direct_bilirubin = float(input("Direct Bilirubin: "))
    tot_proteins = float(input("Total Proteins: "))
    albumin = float(input("Albumin: "))
    ag_ratio = float(input("Albumin/Globulin Ratio: "))
    sgpt = float(input("SGPT: "))
    sgot = float(input("SGOT: "))
    alkphos = float(input("Alkaline Phosphatase: "))

    # Return the data as a dictionary
    return {
        'age': age,
        'gender': gender,
        'tot_bilirubin': tot_bilirubin,
        'direct_bilirubin': direct_bilirubin,
        'tot_proteins': tot_proteins,
        'albumin': albumin,
        'ag_ratio': ag_ratio,
        'sgpt': sgpt,
        'sgot': sgot,
        'alkphos': alkphos
    }

# Main function to run the script
if __name__ == '__main__':
    # Get user input
    user_data = get_input_from_user()

    # Predict liver disease using the trained model
    result = predict_liver_disease(user_data)
    print(result)
