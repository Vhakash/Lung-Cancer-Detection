import tensorflow as tf
import keras
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split



#----------------- STEP 1: D A T A   P R E P A R A T I O N -----------------
def load_lung_cancer_data():
    """
    Load and prepare the lung cancer data for transfer learning

    Returns: X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    """

    print("=" * 50)
    print("LOADING THE LUNG CANCER DETECTION DATA")
    print("=" * 50)

    #Load the preprocessed data
    print("📂 Loading the preprocess data")

    with np.load('preprocessed_data.npz') as data:
        X = data['images']
        y = data['labels']

    # Load class names
    with open('class_names.json', 'r') as f:
        class_names_dict = json.load(f)
    
    print(f"✅ Preprocess data loaded successfully: {X.shape}, {y.shape}")
    print(f"📊 Classes: {class_names_dict}")

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 Original class distribution:")
    for i, count in zip(unique, counts):
        class_name = [k for k, v in class_names_dict.items() if v == i][0]
        print(f"   • Class {i} ({class_name}): {count} images")

    #Data preprocessing
    print("\n🔧 Preprocessing for transfer learning...")
    if X.max() > 1.0:
        X = X.astype('float32') / 255.0
        print("   • Normalized to [0,1] range")
    else:
        X = X.astype('float32')
        print("   • Already Normalized")

    # Calculate Class Weights BEFORE converting to categorical
    print("\n⚖️ Calculating class weights for imbalance...")
    class_labels = np.unique(y)
    y_indices = y  # Keep original integer labels for class weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=y_indices
    )
    class_weights = dict(zip(class_labels, weights))
    print(f"   • Class weights calculated: {class_weights}")

    #convert to categorical 
    y_categorical = keras.utils.to_categorical(y, num_classes = 3)
    print("   • Converted labels to categorical")

    # Split data
    print(f"\n📊 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, 
        stratify=y_indices  # Use original labels for stratification
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=np.argmax(y_temp, axis=1)  # Use categorical labels for stratification
    )
    
    print(f"   • Training: {X_train.shape}")
    print(f"   • Validation: {X_val.shape}")
    print(f"   • Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


# Load the data
X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_lung_cancer_data()




# -------------------------------- STEP 2  : D A T A  A U G M E N T A T I O N ------------------------------------
#to tackle the issue of overfitting and improve model generalization and avoid class imbalance
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1)
])





# -----------------------------STEP - 3 : B U I L D  T R A N S F E R  L E A R N I N G  M O D E L ------------------------------------
print("\n🏗️ Building EfficientNet Transfer Learning Model...")

#load the pre-trained transfer learning model
base_model = EfficientNetB0(
    weights = 'imagenet',
    include_top = False,
    input_shape = (224, 224, 3)
)

#freeze the base model initially
base_model.trainable = False
print(f"   • Base model loaded with {len(base_model.layers)} layers.")
print("   • Base model frozen for feature extraction")

# Create the complete model
model = Sequential([
    data_augmentation,              # Data augmentation
    base_model,                     # Pre-trained feature extractor
    GlobalAveragePooling2D(),       # Convert 2D features to 1D
    Dropout(0.2),                   # Regularization
    Dense(128, activation='relu'),   # Classification head
    Dropout(0.5),
    Dense(3, activation='softmax')   # 3 classes output
])

print(f"   • Model created with {model.count_params():,} total parameters")





#----------------- STEP 4: M O D A L  C O M P I L A T I O N -----------------
# --- Compile Model ---
print("\n⚙️ Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("   • Model compiled for feature extraction phase")
model.summary()




#------------------------STEP 5: S E T   U P   C A L L B A C K S -----------------
# --- Set up callbacks ---
callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        restore_best_weights = True,
        verbose = 1
    ),

    ModelCheckpoint(
        'lung_cancer_transfer_model.h5',
        monitor = 'val_loss',
        save_best_only = True,
        verbose = 1
    ),

    ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.2,
        patience = 3,
        min_lr = 1e-7,
        verbose = 1
    )

]

print("   • Callbacks configured")




#------------------------STEP 6: PHASE 1 - F E A T U R E   E X T R A C T I O N (training) -----------------
print("\n" + "=" * 50)
print("\n🔍 Phase 1 - Feature Extraction...")
print("=" * 50)

history_1 = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs = 15,
    batch_size = 8,
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 'auto'
)

print("✅ Phase 1 completed - Feature extraction training done")
print("   • Model trained for 15 epochs on the training set")
print("   • Validation loss: {:.4f}".format(history_1.history['val_loss'][-1]))




#-----------------------Step 7: PHASE 2 - F I N E - T U N I N G ------------------------------------
print("\n" + "="*50)
print("\n🔍 Phase 2 - Fine Tuning...")
print("=" * 50)

# UNFREEZE the base model
base_model.trainable = True
print(f"   • Base model unfrozen for fine-tuning")

#optional but i am freezing the first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False
print("   • First 100 layers frozen for fine-tuning")

#Recompile with a lower learning rate
model.compile(
    optimizer = Adam(learning_rate = 0.0001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
print(f"  • Model recompiled for fine-tuning with lower learning rate")

#Train for fine tuning
history_2 = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs = 20,
    batch_size = 8,
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 'auto'
)

print("✅ Phase 2 completed - Fine-tuning training done")




#---------------------Step 7: Evaluation Model---------------------
print('\n📊 Evaluating model...')

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 'auto')

print(f"   • Test Accuracy: {test_accuracy:.4f}")
print(f"   • Test Loss: {test_loss:.4f}")

#Make Predictions
predictions = model.predict(X_test)
predicted_class = np.argmax(predictions, axis = 1)

true_classes = np.argmax(y_test, axis = 1)

# Calculate per-class accuracy
from sklearn.metrics import classification_report
print("\n📈 Detailed Classification Report:")
print(classification_report(true_classes,predicted_class))


