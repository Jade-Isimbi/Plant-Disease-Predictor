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
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    file_ext = Path(file.filename).suffix
    if file_ext not in allowed_extensions:
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
    
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
        return jsonify({'success': False, 'error': f'Prediction failed: {error_msg}'}), 500

@app.route('/api/upload/bulk', methods=['POST'])
def bulk_upload_endpoint():
    """Upload multiple images for retraining (no predictions)"""
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': 'No image files provided'}), 400
    
    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
    
    try:
        uploaded_files = []
        errors = []
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # Check file extension
                file_ext = Path(file.filename).suffix
                if file_ext not in allowed_extensions:
                    errors.append(f"{file.filename}: Invalid file type (must be JPG or PNG)")
                    continue
                
                # Save uploaded file first, then check size
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                safe_filename = file.filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"{timestamp}_{safe_filename}"
                filepath = UPLOAD_DIR / filename
                
                file.save(str(filepath))
                
                # Verify file was saved and check size
                if not filepath.exists():
                    errors.append(f"{file.filename}: Failed to save file")
                    continue
                
                file_size = filepath.stat().st_size
                
                if file_size == 0:
                    errors.append(f"{file.filename}: Empty file")
                    try:
                        filepath.unlink()  # Remove empty file
                    except:
                        pass
                    continue
                
                if file_size > 10 * 1024 * 1024:  # 10MB
                    errors.append(f"{file.filename}: File too large (max 10MB, got {file_size / (1024*1024):.2f}MB)")
                    try:
                        filepath.unlink()  # Remove oversized file
                    except:
                        pass
                    continue
                
                uploaded_files.append({
                    'original_name': file.filename,
                    'saved_name': filename,
                    'size': file_size
                })
                print(f"Successfully uploaded: {file.filename} -> {filename} ({file_size / 1024:.2f} KB)")
                    
            except Exception as file_error:
                error_msg = f"Error processing {file.filename}: {str(file_error)}"
                errors.append(error_msg)
                print(f"Upload error: {error_msg}")
                import traceback
                print(traceback.format_exc())
        
        if not uploaded_files and errors:
            return jsonify({
                'success': False,
                'error': 'Failed to upload any images',
                'errors': errors,
                'uploaded': 0,
                'failed': len(errors)
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} image(s)',
            'uploaded': len(uploaded_files),
            'failed': len(errors),
            'files': uploaded_files,
            'errors': errors if errors else None
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Bulk upload error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Bulk upload failed: {error_msg}'
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch_endpoint():
    """Predict plant disease from multiple uploaded images"""
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': 'No image files provided'}), 400
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    try:
        results = []
        errors = []
        
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # Check file extension
                allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                file_ext = Path(file.filename).suffix
                if file_ext not in allowed_extensions:
                    errors.append(f"{file.filename}: Invalid file type")
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
            except Exception as file_error:
                error_msg = f"Error processing {file.filename}: {str(file_error)}"
                errors.append(error_msg)
                print(error_msg)
        
        if not results and errors:
            return jsonify({
                'success': False,
                'error': 'Failed to process any images',
                'errors': errors
            }), 500
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'errors': errors if errors else None
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Batch prediction error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Batch prediction failed: {error_msg}'
        }), 500

@app.route('/api/upload/count', methods=['GET'])
def get_upload_count():
    """Get count of uploaded files ready for retraining"""
    try:
        upload_files = list(UPLOAD_DIR.glob('*.JPG')) + list(UPLOAD_DIR.glob('*.jpg')) + list(UPLOAD_DIR.glob('*.png'))
        return jsonify({
            'success': True,
            'count': len(upload_files)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'count': 0
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining with uploaded data"""
    try:
        # Check if new data exists in uploads
        upload_files = list(UPLOAD_DIR.glob('*.JPG')) + list(UPLOAD_DIR.glob('*.jpg')) + list(UPLOAD_DIR.glob('*.png'))
        
        if not upload_files:
            return jsonify({'success': False, 'error': 'No new data found for retraining'}), 400
        
        print(f"Starting retraining with {len(upload_files)} uploaded files...")
        
        # Check if train and test directories exist
        if not TRAIN_DIR.exists() or not TEST_DIR.exists():
            error_msg = f"Training or test directory not found. Train: {TRAIN_DIR.exists()}, Test: {TEST_DIR.exists()}"
            print(f"Error: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 500

        print("Creating data generators with optimized settings for faster training...")
        try:
            # Use smaller image size and larger batch size for faster training
            train_generator, test_generator = create_data_generators(
                TRAIN_DIR, TEST_DIR, 
                img_size=160,  # Reduced from 224 for faster processing
                batch_size=64   # Increased from 32 for faster processing
            )
            if train_generator is None or test_generator is None:
                raise ValueError("Failed to create data generators")
        except Exception as gen_error:
            error_msg = f"Failed to create data generators: {str(gen_error)}"
            print(f"Error: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 500
        
        print("Getting class names...")
        try:
            num_classes = len(get_class_names(TRAIN_DIR))
            if num_classes == 0:
                error_msg = "No classes found in training data"
                print(f"Error: {error_msg}")
                return jsonify({'success': False, 'error': error_msg}), 500
        except Exception as class_error:
            error_msg = f"Failed to get class names: {str(class_error)}"
            print(f"Error: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 500
        
        print(f"Creating model with {num_classes} classes (optimized for fast retraining)...")
        try:
            # Use smaller image size for faster training
            new_model = create_model(num_classes, img_size=160)
            if new_model is None:
                raise ValueError("Failed to create model")
        except Exception as model_error:
            error_msg = f"Failed to create model: {str(model_error)}"
            print(f"Error: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 500
        
        print("Starting model training with optimized settings for speed...")
        # Train model with optimized settings for faster retraining
        try:
            # Calculate steps per epoch (limit to avoid processing all data)
            steps_per_epoch = min(100, len(train_generator))  # Max 100 steps per epoch
            validation_steps = min(50, len(test_generator))   # Max 50 validation steps
            
            history = train_model(
                new_model, 
                train_generator, 
                test_generator,
                epochs=2,  # Reduced from 5 to 2 for faster retraining
                model_save_path=str(MODEL_PATH),
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )
            if history is None:
                raise ValueError("Training returned no history")
        except Exception as train_error:
            error_msg = f"Training failed: {str(train_error)}"
            print(f"Error: {error_msg}")
            import traceback
            print(traceback.format_exc())
            return jsonify({'success': False, 'error': error_msg}), 500
        
        print("Training complete. Reloading model...")
        # Reload model
        load_model_if_exists()
        
        # Extract accuracy safely
        final_accuracy = 0.0
        epochs_trained = 0
        if history and hasattr(history, 'history'):
            if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
                final_accuracy = float(history.history['val_accuracy'][-1])
            if 'accuracy' in history.history:
                epochs_trained = len(history.history['accuracy'])
        
        print(f"Retraining complete. Final accuracy: {final_accuracy:.2%}")
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'final_accuracy': final_accuracy,
            'epochs_trained': epochs_trained
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Retraining error: {error_msg}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Retraining failed: {error_msg}'
        }), 500

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

