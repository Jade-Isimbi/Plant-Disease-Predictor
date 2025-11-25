"""
Prediction functions for plant disease classification
"""
import numpy as np
from tensorflow.keras.preprocessing import image
from .preprocessing import preprocess_image


def predict_image(image_path, model, class_names, img_size=224):
    """
    Predict the class of a single image
    
    Args:
        image_path: Path to the image file
        model: Trained Keras model
        class_names: List of class names
        img_size: Target image size (default: 224)
    
    Returns:
        predicted_class: Name of predicted class
        confidence: Confidence score (0-1)
        all_predictions: Dictionary of all class predictions
    """
    # Preprocess image
    img_array = preprocess_image(image_path, img_size)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx])
    predicted_class = class_names[predicted_idx]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    all_predictions = {
        class_names[i]: float(predictions[0][i]) 
        for i in top_5_indices
    }
    
    return predicted_class, confidence, all_predictions


def predict_batch(image_paths, model, class_names, img_size=224):
    """
    Predict classes for multiple images
    
    Args:
        image_paths: List of paths to image files
        model: Trained Keras model
        class_names: List of class names
        img_size: Target image size (default: 224)
    
    Returns:
        List of predictions, each containing (predicted_class, confidence, all_predictions)
    """
    results = []
    for img_path in image_paths:
        try:
            pred_class, confidence, all_preds = predict_image(
                img_path, model, class_names, img_size
            )
            results.append({
                'image_path': str(img_path),
                'predicted_class': pred_class,
                'confidence': confidence,
                'all_predictions': all_preds
            })
        except Exception as e:
            results.append({
                'image_path': str(img_path),
                'error': str(e)
            })
    
    return results

