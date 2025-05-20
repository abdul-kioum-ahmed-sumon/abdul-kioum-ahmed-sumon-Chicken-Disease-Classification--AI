from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline
import shutil
import logging
from datetime import datetime
import json
from pathlib import Path
import time


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create required directories
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
SAMPLES_FOLDER = os.path.join(STATIC_FOLDER, 'samples')
DATASET_PATH = "research/Chicken-Coccidiosis-Dataset"
HISTORY_FILE = 'prediction_history.json'

# Default sample image URLs (as fallback)
DEFAULT_SAMPLES = {
    'healthy1': 'https://i.ibb.co/Jt8MH9j/healthy1.jpg',
    'healthy2': 'https://i.ibb.co/wSxFC3y/healthy2.jpg',
    'coccidiosis1': 'https://i.ibb.co/0mjvXf6/coccidiosis1.jpg',
    'coccidiosis2': 'https://i.ibb.co/VxBxrRh/coccidiosis2.jpg'
}

# Create required directories
for folder in [UPLOAD_FOLDER, STATIC_FOLDER, SAMPLES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def setup_sample_images():
    """Set up sample images from either dataset or download from default URLs"""
    # First try to copy from dataset
    dataset_samples = False
    if os.path.exists(DATASET_PATH):
        try:
            # Copy healthy samples
            healthy_path = os.path.join(DATASET_PATH, "Healthy")
            if os.path.exists(healthy_path):
                for i, file in enumerate(os.listdir(healthy_path)[:2]):
                    src = os.path.join(healthy_path, file)
                    dst = os.path.join(SAMPLES_FOLDER, f"healthy{i+1}.jpg")
                    shutil.copy2(src, dst)
            
            # Copy coccidiosis samples
            coccidiosis_path = os.path.join(DATASET_PATH, "Coccidiosis")
            if os.path.exists(coccidiosis_path):
                for i, file in enumerate(os.listdir(coccidiosis_path)[:2]):
                    src = os.path.join(coccidiosis_path, file)
                    dst = os.path.join(SAMPLES_FOLDER, f"coccidiosis{i+1}.jpg")
                    shutil.copy2(src, dst)
            
            # Check if all sample images were copied
            required_samples = ['healthy1.jpg', 'healthy2.jpg', 'coccidiosis1.jpg', 'coccidiosis2.jpg']
            dataset_samples = all(os.path.exists(os.path.join(SAMPLES_FOLDER, sample)) for sample in required_samples)
        except Exception as e:
            print(f"Error copying dataset samples: {str(e)}")
            dataset_samples = False
    
    # If dataset samples failed, use default URLs
    if not dataset_samples:
        try:
            import requests
            for name, url in DEFAULT_SAMPLES.items():
                target_path = os.path.join(SAMPLES_FOLDER, f"{name}.jpg")
                if not os.path.exists(target_path):
                    response = requests.get(url, timeout=10)  # Add 10 second timeout
                    if response.status_code == 200:
                        with open(target_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        print(f"Failed to download sample image {name} from {url}")
        except Exception as e:
            print(f"Error downloading sample images: {str(e)}")

# Set up sample images
setup_sample_images()

app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = os.path.join(UPLOAD_FOLDER, "inputImage.jpg")
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


def save_prediction_history(prediction_data):
    """Save prediction results to history file"""
    try:
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        
        # Add timestamp and limit history to last 100 predictions
        prediction_data['timestamp'] = datetime.now().isoformat()
        history.append(prediction_data)
        history = history[-100:]
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving prediction history: {str(e)}")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        start_time = time.time()
        logger.info("Starting new prediction request")
        
        if not request.json or 'image' not in request.json:
            logger.warning("No image data provided in request")
            return jsonify({"error": "No image data provided"}), 400
            
        image = request.json['image']
        if not image:
            logger.warning("Empty image data received")
            return jsonify({"error": "Empty image data"}), 400
            
        try:
            # Ensure the upload directory exists
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
                
            # Clean up old image if it exists
            if os.path.exists(clApp.filename):
                os.remove(clApp.filename)
                
            decodeImage(image, clApp.filename)
            
            if not os.path.exists(clApp.filename):
                logger.error("Failed to save decoded image")
                return jsonify({"error": "Failed to save decoded image"}), 500
                
        except Exception as decode_error:
            logger.error(f"Error processing image: {str(decode_error)}")
            return jsonify({"error": f"Error processing image: {str(decode_error)}"}), 400
            
        try:
            # Perform prediction
            result = clApp.classifier.predict()
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            # Add processing time to result
            if isinstance(result, list) and len(result) > 0:
                result[0]['processing_time'] = processing_time
                
                # Save prediction to history
                save_prediction_history(result[0])
                
                logger.info(f"Prediction completed successfully in {processing_time}s")
            
            return jsonify(result)
            
        except Exception as predict_error:
            logger.error(f"Prediction failed: {str(predict_error)}")
            return jsonify({
                "error": "Prediction failed",
                "details": str(predict_error),
                "success": False
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Unexpected error",
            "details": str(e),
            "success": False
        }), 500


@app.route("/history", methods=['GET'])
@cross_origin()
def get_prediction_history():
    """Endpoint to retrieve prediction history"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return jsonify(history)
        return jsonify([])
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        return jsonify({"error": "Failed to retrieve prediction history"}), 500


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=5000, debug=True)

