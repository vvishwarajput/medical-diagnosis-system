import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('new_liver_data.csv')

# Step 1: Inspect the 'gender' column for any unexpected entries
print("\nChecking unique values in the 'gender' column:")
print(df['gender'].unique())  # Check the unique entries to spot any mixed or incorrect data

# Step 2: Clean the 'gender' column by removing any non-standard entries
# We'll strip spaces and check if there are any mixed values like 'FemaleMale' that shouldn't be there
df['gender'] = df['gender'].str.strip()

# If there are any non-standard values, handle them
# Replace any unexpected values with 'Unknown' or remove them
df = df[df['gender'].isin(['Male', 'Female'])]  # Keep only 'Male' and 'Female'

# Step 3: Map 'Male' to 0 and 'Female' to 1
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Step 4: Handle missing values in other columns
df.fillna(df.mean(), inplace=True)  # Replace missing values with column mean for numerical data

# Step 5: Check data types and ensure the dataset is clean
print("\nData types of each column:")
print(df.dtypes)

# Step 6: One-hot encode any remaining categorical columns (if any)
# For now, 'gender' has been encoded, so let's focus on the rest of the data
# Define features and target variable
X = df.drop('is_patient', axis=1)  # All columns except the target 'is_patient'
y = df['is_patient']  # The target variable

# Step 7: Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature scaling for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Initialize the classifier (Random Forest)
model = RandomForestClassifier(random_state=42)

# Step 10: Train the model
model.fit(X_train_scaled, y_train)

# Step 11: Predict on the test set
y_pred = model.predict(X_test_scaled)

# Step 12: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Save the trained model and scaler for later use
joblib.dump(model, 'liver_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
