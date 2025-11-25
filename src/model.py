"""
Model creation and training functions for plant disease classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path


def create_model(num_classes, img_size=224):
    """
    Create a plant disease classification model using transfer learning
    
    Args:
        num_classes: Number of classes to classify
        img_size: Input image size (default: 224)
    
    Returns:
        Compiled Keras model
    """
    # Use MobileNetV2 as base model
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, train_generator, test_generator, epochs=10, 
                model_save_path='models/plant_disease_model.h5'):
    """
    Train the model
    
    Args:
        model: Keras model to train
        train_generator: Training data generator
        test_generator: Test/validation data generator
        epochs: Number of epochs (default: 10)
        model_save_path: Path to save the best model
    
    Returns:
        Training history
    """
    # Create model directory if it doesn't exist
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        str(model_save_path),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history


def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)



