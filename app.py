from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2

# Initialize the Flask application
app = Flask(__name__)

# Set directory to save uploaded images
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


# Load the pre-trained pneumonia detection model
model = load_model('/home/keyan/model/pneumonia_detection_model_18.h5')

# Define image size based on training data
IMG_SIZE = 150

# Function to calculate affected lung area based on simple thresholding
def calculate_affected_area(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))

    # Apply binary threshold
    _, binary_img = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)

    # Create a color mask to highlight affected areas
    colored_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    colored_img[binary_img == 255] = [0, 0, 255]  # Highlight affected area in red

    # Save the highlighted image
    highlighted_image_filename = 'highlighted_' + os.path.basename(img_path)
    highlighted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], highlighted_image_filename)

    # Print to debug
    print(f"Saving highlighted image at: {highlighted_image_path}")

    cv2.imwrite(highlighted_image_path, colored_img)
    
    # Check if saving was successful
    if not os.path.isfile(highlighted_image_path):
        print("Error: Highlighted image was not saved!")
        if os.path.isfile(highlighted_image_path):
            print("Image saved successfully.")
        else:
            print("Image not found after saving.")


    # Calculate affected area
    total_pixels = binary_img.size
    affected_pixels = np.sum(binary_img == 255)
    affected_percentage = (affected_pixels / total_pixels) * 100

    return affected_percentage, highlighted_image_filename




# Function to predict if pneumonia is present and calculate the affected area
def predict_pneumonia_and_calculate_area(model, img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict pneumonia
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        pneumonia_result = "Pneumonia Detected"
        affected_area, highlighted_image_path = calculate_affected_area(img_path)  # Unpacking
        result = f"{pneumonia_result}. Lung affected area: {affected_area:.2f}%"
    else:
        result = "No Pneumonia Detected"
        highlighted_image_path = None
    
    return result, highlighted_image_path

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Perform pneumonia detection and affected area calculation
        result, highlighted_image_filename = predict_pneumonia_and_calculate_area(model, file_path)
        
        # Construct the full path for the highlighted image
        highlighted_image_url = f"/static/uploads/{highlighted_image_filename}" if highlighted_image_filename else None
        
        # Return result to the user
        return render_template('result.html', result=result, highlighted_image_url=highlighted_image_url)

if __name__ == "__main__":
    app.run(debug=True)
