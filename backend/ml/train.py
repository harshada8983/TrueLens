import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. Configuration & Hyperparameters ---
# WARNING: Change this string to the exact Windows path of your dataset!
DATA_DIR = "C:/TrueLens/backend/data/train_clean" 
MODEL_SAVE_PATH = "../models/truelens_efficientnet.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50 # The Marathon Setting

print(f"Initializing TrueLens Engine Training Pipeline...")
print(f"Target Dataset: {DATA_DIR}")

# --- 2. Data Loading & Augmentation ---
# We use 80% of data for training, 20% for testing/validation
datagen = ImageDataGenerator(
    validation_split=0.2, 
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- 3. Neural Network Architecture (EfficientNetB0) ---
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True # Freeze base model initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    # --- THE UPGRADE ---
    # We are dropping 70% of the neurons every batch. 
    # This prevents the AI from memorizing the data and forces it to learn real forensics.
    Dropout(0.7), 
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. The Iron-Clad Callbacks ---
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# --- 5. THE ENGINE START (model.fit) ---
print("\n[ TENSORS READY. INITIATING NEURAL INFERENCE RUN... ]\n")

# HERE IT IS: The command that runs the marathon
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("\n[ TRAINING MARATHON COMPLETE. BEST MODEL SAVED. ]")