"""
Data preprocessing functions for plant disease classification
"""
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(train_dir, test_dir, img_size=224, batch_size=32):
    """
    Create data generators for training and testing
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        img_size: Target image size (default: 224)
        batch_size: Batch size (default: 32)
    
    Returns:
        train_generator: Training data generator with augmentation
        test_generator: Test data generator (no augmentation)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescale for test (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator


def preprocess_image(image_path, img_size=224):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image file
        img_size: Target image size (default: 224)
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


def get_class_names(data_dir):
    """
    Get sorted list of class names from data directory
    
    Args:
        data_dir: Path to data directory containing class folders
    
    Returns:
        Sorted list of class names
    """
    data_path = Path(data_dir)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    return class_names



