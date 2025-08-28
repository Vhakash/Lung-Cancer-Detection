import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json

# --- Load Data ---
print("Loading preprocessed data...")
with np.load('preprocessed_data.npz') as data:
    X = data['images']
    y = data['labels']

with open('class_names.json', 'r') as f:
    class_names_dict = json.load(f)

print(f"Data loaded: {X.shape}, {y.shape}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Original class distribution:")
for i, count in zip(unique, counts):
    print(f"Class {i}: {count} images")

# --- Create Balanced Dataset ---
print("\n--- Creating Balanced Dataset ---")

# Find the minimum class size
min_samples = min(counts)
print(f"Using {min_samples} samples per class for balance")

# Sample equal numbers from each class
balanced_X = []
balanced_y = []

for class_id in unique:
    # Get indices for this class
    class_indices = np.where(y == class_id)[0]
    
    # Randomly sample min_samples indices
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(class_indices, min_samples, replace=False)
    
    # Add to balanced dataset
    balanced_X.extend(X[sampled_indices])
    balanced_y.extend(y[sampled_indices])

# Convert to numpy arrays
X_balanced = np.array(balanced_X)
y_balanced = np.array(balanced_y)

print(f"Balanced dataset: {X_balanced.shape}, {y_balanced.shape}")

# Check new distribution
unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
print("Balanced class distribution:")
for i, count in zip(unique_balanced, counts_balanced):
    print(f"Class {i}: {count} images")

# --- Train/Test Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_balanced
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# --- Build Model ---
print("\n--- Building Model ---")
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# --- Train Model ---
print("\n--- Training Balanced Model ---")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose="auto"  # type: ignore
)

# --- Save Improved Model ---
model.save('lung_cancer_detector_balanced.h5')
print("Balanced model saved as 'lung_cancer_detector_balanced.h5'")

# --- Test the Model ---
print("\n--- Testing Model on Validation Set ---")
val_predictions = model.predict(X_val)
val_pred_classes = np.argmax(val_predictions, axis=1)

# Print some predictions
print("Sample predictions:")
for i in range(min(10, len(X_val))):
    true_class = y_val[i]
    pred_class = val_pred_classes[i]
    confidence = val_predictions[i][pred_class]
    print(f"Image {i}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.3f}")
