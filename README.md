# Plant Disease Classification - ML Pipeline

A complete ML pipeline for classifying plant diseases from leaf images using transfer learning with MobileNetV2.

##  Video Demo

**Link:** https://youtu.be/HzkSPL5lxzI

##  Web app URL

**Link:** https://plant-disease-predictor-nqho.onrender.com/

##  Project Description

This system identify plant diseases from leaf images and continously improve through the bulk upload which will retrain the model and improve it. This project uses the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) from Kaggle.

The main features are:

- Real-time disease prediction from images
- Interactive web dashboard
- Model retraining with new data 
- RESTful API for integration
- 38 disease categories supported

## Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/Jade-Isimbi/Plant-Disease-Predictor
cd Plant-Disease-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Run application
python api/app.py
```

Access at: http://localhost:5000

##  Notebook

**File:** `notebook/PlantPredictor.ipynb`

### Preprocessing Steps
- Image resizing to 160x160 pixels
- Data augmentation (rotation, shifts, flips, zoom)
- Train/test split (80/20)
- Normalization 

### Model Training
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Epochs**: 2 
- **Batch Size**: 64
- **Image Size**: 160x160
- **Training Time**: 2-3 minutes 

### Prediction Functions
```python
from src.prediction import predict_image

predicted_class, confidence, all_predictions = predict_image(
    image_path='data/test',
    model=model,
    class_names=class_names
)
```

## Load Testing Results

**Tool:** Locust

**Run Tests:**
```bash
pip install locust
locust -f tests/locustfile.py --host=http://localhost:5000
```

**Overall performance:**

For 50 concurrent users . These were the results:
- 4.8 requests per second (RPS)
- 0% failure rate
- Total requests: 6,961
  


##  Model File

**Location:** `models/plant_disease_model.h5`

**Format:** Keras H5 (.h5)

**Details:**
- Architecture: MobileNetV2
- Input: 160x160x3 RGB images
- Output: 38 classes
- Size: ~15-20 MB
- Accuracy: ~92% validation

**Load Model:**
```python
from src.model import load_model
model = load_model('models/plant_disease_model.h5')
```

##  API Endpoints

- `GET /api/health` - Health check
- `GET /api/stats` - Model statistics
- `POST /api/predict` - Single image prediction
- `POST /api/upload/bulk` - Bulk image upload
- `POST /api/retrain` - Trigger model retraining
- `GET /api/upload/count` - Count uploaded files

## Project Structure

```
Plant-Disease-Predictor/
├── api/app.py              # Flask API
├── models/                 # Trained model (.h5)
├── notebook/               # Jupyter notebook
├── src/                    # Source code
│   ├── model.py
│   ├── preprocessing.py
│   └── prediction.py
├── data/                   # Dataset
│   ├── train/
│   └── test/
└── tests/                  # Load testing
```

## Technologies

- Python 3.9+
- TensorFlow 2.x
- Flask
- MobileNetV2
- Locust (load testing)



