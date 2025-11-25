"""
Flask API for Plant Disease Classification
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# Import model functions
import sys
BASE_DIR_PARENT = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR_PARENT))
from src.model import load_model, create_model, train_model
from src.preprocessing import get_class_names, create_data_generators, preprocess_image
from src.prediction import predict_image, predict_batch

app = Flask(__name__, 
            template_folder=Path(__file__).parent.parent / 'templates',
            static_folder=Path(__file__).parent.parent / 'static')
CORS(app)

# Configure max file size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'plant_disease_model.h5'
TRAIN_DIR = BASE_DIR / 'data' / 'train'
TEST_DIR = BASE_DIR / 'data' / 'test'
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Global variables
model = None
class_names = None
model_loaded_at = None

def load_model_if_exists():
    """Load model if it exists"""
    global model, class_names, model_loaded_at
    if MODEL_PATH.exists():
        try:
            model = load_model(str(MODEL_PATH))
            class_names = get_class_names(TRAIN_DIR)
            model_loaded_at = datetime.now().isoformat()
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model not found at {MODEL_PATH}")

# Load model on startup
load_model_if_exists()

# Error handler for file size limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_loaded_at': model_loaded_at,
        'num_classes': len(class_names) if class_names else 0
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict plant disease from uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    file_ext = Path(file.filename).suffix
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
    
    try:
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOAD_DIR / filename
        file.save(str(filepath))
        
        # Predict
        predicted_class, confidence, all_predictions = predict_image(
            str(filepath), model, class_names
        )
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': all_predictions,
            'image_path': filename
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {error_msg}'}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch_endpoint():
    """Predict plant disease from multiple uploaded images"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            filepath = UPLOAD_DIR / filename
            file.save(str(filepath))
            
            # Predict
            predicted_class, confidence, all_predictions = predict_image(
                str(filepath), model, class_names
            )
            
            results.append({
                'filename': file.filename,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': all_predictions
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining with uploaded data"""
    try:
        # Check if new data exists in uploads
        upload_files = list(UPLOAD_DIR.glob('*.JPG')) + list(UPLOAD_DIR.glob('*.jpg')) + list(UPLOAD_DIR.glob('*.png'))
        
        if not upload_files:
            return jsonify({'error': 'No new data found for retraining'}), 400
        
        # For now, retrain with existing train/test data
        # In production, you'd merge uploaded data with training data
        train_generator, test_generator = create_data_generators(
            TRAIN_DIR, TEST_DIR
        )
        
        num_classes = len(get_class_names(TRAIN_DIR))
        new_model = create_model(num_classes)
        
        # Train model
        history = train_model(
            new_model, 
            train_generator, 
            test_generator,
            epochs=5,  # Fewer epochs for retraining
            model_save_path=str(MODEL_PATH)
        )
        
        # Reload model
        load_model_if_exists()
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'final_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['accuracy'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """Get model and dataset statistics"""
    if model is None or class_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Count images per class (handle missing directories gracefully)
    train_counts = {}
    test_counts = {}
    total_train = 0
    total_test = 0
    
    # Only count if directories exist
    if TRAIN_DIR.exists() and TEST_DIR.exists():
        for class_name in class_names:
            train_path = TRAIN_DIR / class_name
            test_path = TEST_DIR / class_name
            
            train_count = 0
            test_count = 0
            
            if train_path.exists():
                train_images = list(train_path.glob('*.JPG')) + list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
                train_count = len(train_images)
            
            if test_path.exists():
                test_images = list(test_path.glob('*.JPG')) + list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
                test_count = len(test_images)
            
            train_counts[class_name] = train_count
            test_counts[class_name] = test_count
            total_train += train_count
            total_test += test_count
    
    return jsonify({
        'num_classes': len(class_names) if class_names else 0,
        'total_train_images': total_train,
        'total_test_images': total_test,
        'class_distribution': {
            'train': train_counts,
            'test': test_counts
        },
        'model_loaded_at': model_loaded_at
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)

