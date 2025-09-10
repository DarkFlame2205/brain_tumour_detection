import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define dataset and training paths
DATASET_PATH = "dataset"  # This folder contains all MRI images
TRAINING_PATH = "training"  # Only these images will be used for training
MODEL_PATH = "models"  # Folder to save the trained model
CATEGORIES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = 128  # Reduced size to save memory

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

def load_data(data_path):
    X, y = [], []
    for category in CATEGORIES:
        category_path = os.path.join(data_path, category)
        label = CATEGORIES.index(category)

        if not os.path.exists(category_path):
            continue  # Skip if category folder does not exist

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0  # Normalize & use float32 to reduce memory
            X.append(img)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Load dataset (all images)
X_dataset, y_dataset = load_data(DATASET_PATH)

# Load training set (subset of dataset)
X_train, y_train = load_data(TRAINING_PATH)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define CNN model with MaxPooling2D
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  
    MaxPooling2D(2,2),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),  # Lower dropout to retain learning
    Dense(len(CATEGORIES), activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model with smaller batch size to save memory
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save model in the models folder
model.save(os.path.join(MODEL_PATH, "brain_tumor_model.h5"))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
