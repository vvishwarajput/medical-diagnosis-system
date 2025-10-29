import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Set the correct path to your dataset
dataset_path = "train_lung"

# Image dimensions
IMG_SIZE = 128

# Label encoding
label_map = {
    'bacterial neumonia': 0,
    'corona virus': 1,
    'normal': 2
}

# Lists to hold images and labels
images = []
labels = []

# Load and preprocess images
for label_name, label_idx in label_map.items():
    folder_path = os.path.join(dataset_path, label_name)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label_idx)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# One-hot encode the labels
y = to_categorical(y, num_classes=3)

# Standardize the data (Optional for images, but good practice for neural nets)
scaler = StandardScaler()
X_flattened = X.reshape(-1, IMG_SIZE * IMG_SIZE * 3)  # Flatten images into 1D array for scaling
X_scaled = scaler.fit_transform(X_flattened)  # Standardize the flattened images
X_scaled = X_scaled.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Reshape back to the original image shape

# Split into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Print the shapes for confirmation
print("✅ Data Preprocessing Completed!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Save the scaler for future use (for inference/scaling new images)
import joblib
joblib.dump(scaler, 'scaler.pkl')
