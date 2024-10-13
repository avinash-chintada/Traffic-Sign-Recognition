import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model.h5'
UPLOAD_FOLDER = 'uploads'

# Load your trained model
model = load_model(MODEL_PATH)
model.summary()  # Print the model summary

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocessing(img):
    try:
        if len(img.shape) == 3:  # Check if the image has 3 channels (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            print("Grayscale conversion successful.")
        
        img = img.astype(np.uint8)  # Ensure the image is 8-bit
        img = cv2.equalizeHist(img)  # Apply histogram equalization
        print("Histogram equalization successful.")
        
        img = img / 255.0  # Normalize the image
        img = np.reshape(img, (32, 32, 1))  # Reshape to (32, 32, 1)
        print(f"Preprocessed image shape: {img.shape}")
        return img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 
        'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 
        'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 
        'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
        'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    
    return class_names[classNo] if classNo < len(class_names) else "Unknown Sign"

def model_predict(img_path, model):
    try:
        print(f"Loading image from path: {img_path}")
        img = image.load_img(img_path, target_size=(32, 32))  # Load and resize image
        img = image.img_to_array(img)
        img = preprocessing(img)
        
        if img is None:
            print("Image preprocessing failed.")
            return "Error in image preprocessing"
        
        # Add batch dimension and ensure the image is in float32 format
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 32, 32, 1)
        img = img.astype('float32')  # Ensure the image is of type float32
        print(f"Image ready for prediction with shape: {img.shape}")
        
        # Predict using the model
        predictions = model.predict(img)
        print(f"Raw predictions: {predictions}")  # Log raw predictions
        
        classIndex = np.argmax(predictions, axis=-1)
        print(f"Predicted class index: {classIndex}")
        
        preds = getClassName(classIndex[0])
        print(f"Prediction successful: {preds}")
        
        return preds
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        print("No file part in the request.")
        return jsonify(predicted_class='No file part')

    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return jsonify(predicted_class='No selected file')

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(f"File saved successfully at: {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify(predicted_class='Error in saving file')

        # Make prediction
        preds = model_predict(file_path, model)
        return jsonify(predicted_class=preds)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
