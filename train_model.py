import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import keras
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
PROCESSED_DATA_FILE = 'preprocessed_data.npz'
CLASS_NAMES_FILE = 'class_names.json'
MODEL_OUTPUT_FILE = 'lung_cancer_detector.h5'
PLOT_OUTPUT_FILE = 'training_history.png'

# --- 1. Load the Preprocessed Data ---

print("Loading preprocessed data...")
with np.load(PROCESSED_DATA_FILE) as data:
    X = data['images']
    y = data['labels']

with open(CLASS_NAMES_FILE, 'r') as f:
    class_names = json.load(f)

print(f"Data loaded successfully!")
print(f"Images shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of classes: {len(class_names)}")

# Reduce memory usage by using a smaller subset for training (optional)
# If you still get memory errors, uncomment the next few lines:
# print("Using subset of data to reduce memory usage...")
# subset_size = 200  # Use only 200 images
# X = X[:subset_size]
# y = y[:subset_size]
# print(f"Reduced to {subset_size} images")

print(f"Data loaded successfully.")
print(f"Images (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")
print(f"Number of classes: {len(class_names)}")










# --- 2. Split the Data ---
print("\n--- Phase 2: Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Then, split the temporary set into validation (10%) and testing (10%)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")










# --- 3. Build the Model ---
print("\n--- Phase 3: Building the Model ---")
# Build a simple CNN model instead of EfficientNet to avoid pre-trained weight issues

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
    Dense(len(class_names), activation='softmax')
])

print("Model architecture:")
model.summary()

# --- Compile the Model ---
print("\n--- Compiling the Model ---")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully!")

# --- 4. Train the Model ---
print("\n--- Phase 4: Training the Model ---")
# Define callbacks for training
# EarlyStopping will stop training if the validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor = 'val_loss', patience = 5, restore_best_weights = True
)
# ModelCheckpoint will save the best version of the model
model_checkpoint = ModelCheckpoint(
    MODEL_OUTPUT_FILE,
    monitor='val_loss',
    save_best_only=True
)

#Train the model
history = model.fit(
    X_train, y_train,
    epochs = 10, # Reduced epochs for faster training
    batch_size = 8, # Smaller batch size to reduce memory usage
    validation_data = (X_val, y_val),
    callbacks = [early_stopping, model_checkpoint],
    verbose = "auto"  # type: ignore
)

























# --- 5. Evaluate the Model ---
print("\n--- Phase 5: Evaluating the Model ---")
# Load the best model saved by ModelCheckpoint
model.load_weights(MODEL_OUTPUT_FILE)

# Evaluate the model on the unseen test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")


# --- 6. Visualize Training History ---
print("\n--- Phase 6: Visualizing Training History ---")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plot to a file
plt.savefig(PLOT_OUTPUT_FILE)
print(f"Training history plot saved to '{PLOT_OUTPUT_FILE}'")
plt.show()

print("\nTraining complete. The best model has been saved.")