"""
Locust load testing script for Plant Disease Classification API
"""
from locust import HttpUser, task, between
import random
import os
from pathlib import Path


class PlantDiseaseUser(HttpUser):
    """Simulate user behavior for load testing"""
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Check if API is healthy
        self.client.get("/api/health")
    
    @task(3)
    def health_check(self):
        """Check API health (high frequency)"""
        self.client.get("/api/health")
    
    @task(2)
    def get_stats(self):
        """Get model statistics"""
        self.client.get("/api/stats")
    
    @task(1)
    def predict_image(self):
        """Predict plant disease from image"""
        # Use a sample image from test data
        test_dir = Path("data/test")
        if test_dir.exists():
            # Find a random test image
            all_images = []
            for class_dir in test_dir.iterdir():
                if class_dir.is_dir():
                    images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    all_images.extend(images)
            
            if all_images:
                # Pick a random image
                image_path = random.choice(all_images)
                
                with open(image_path, 'rb') as f:
                    files = {'image': (image_path.name, f, 'image/jpeg')}
                    self.client.post("/api/predict", files=files, name="/api/predict")
    
    @task(1)
    def view_homepage(self):
        """View the main UI"""
        self.client.get("/")


# Configuration for load testing scenarios
class QuickTestUser(PlantDiseaseUser):
    """Quick test with fewer users"""
    wait_time = between(0.5, 1.5)


class StressTestUser(PlantDiseaseUser):
    """Stress test with more aggressive requests"""
    wait_time = between(0.1, 0.5)




