from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import uuid
from flask_mail import Mail, Message
from flask import send_from_directory

app = Flask(__name__)

# Load the model once when the app starts
try:
    model = load_model('best_advanced_model.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the class names corresponding to the labels
class_names = [
    'Banana Black Sigatoka Disease', 'Banana Bract Mosaic Virus Disease', 
    'Banana Healthy Leaf', 'Banana Insect Pest Disease', 
    'Banana Moko Disease', 'Banana Panama Disease', 'Banana Yellow Sigatoka Disease'
]

# Path for uploading files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to validate image file extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_email_password'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'  # Default sender email

mail = Mail(app)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the disease detection page
@app.route('/disease-detection')
def disease_detection():
    return render_template('disease-detection.html')

# Route for the disease information page
@app.route('/disease-info')
def disease_info():
    return render_template('disease-information.html')

# Route for the remedy and prevention page
@app.route('/remedy-prevention')
def remedy_prevention():
    return render_template('remedy-prevention.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for the Prediction page
@app.route('/prediction-result')
def prediction_result():
    prediction = request.args.get('prediction')
    image_url = request.args.get('image_url')
    return render_template('prediction-result.html', prediction=prediction, image_url=image_url)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only jpg, jpeg, png, and gif are allowed.'}), 400
    
    try:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        print(f"File saved at: {file_path}")
    except Exception as e:
        print(f"File save error: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    try:
        img = Image.open(file_path)
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Image processing error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class]
        confidence_score = predictions[0][predicted_class] * 100
        print(f"Prediction: {predicted_class_name}, Confidence: {confidence_score:.2f}%")
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

    image_url = f'/uploads/{unique_filename}'

    return jsonify({
        'prediction': predicted_class_name,
        'confidence': f"{confidence_score:.2f}%",  # Now included
        'image_url': image_url
    })

@app.route('/remedies')
def remedies():
    disease = request.args.get('disease')
    
    remedies = {
        'Banana Black Sigatoka Disease': [
            'Use systemic fungicides like azoxystrobin or chlorothalonil.',
            'Prune and remove infected leaves to prevent the spread.',
            'Maintain proper irrigation and avoid over-watering.',
            'Ensure good air circulation around the banana plant.'
        ],
        'Banana Bract Mosaic Virus Disease': [
            'There is no cure for the virus. Remove and destroy infected plants immediately.',
            'Use insecticides to control aphid vectors that spread the virus.',
            'Maintain strict quarantine procedures to prevent the spread of infected plants.',
            'Use virus-resistant banana varieties if available.'
        ],
        'Banana Healthy Leaf': [
            'Ensure proper watering and sunlight.',
            'Regularly check for pests and diseases.',
            'Maintain soil fertility with organic fertilizers.',
            'Prune old or damaged leaves to encourage new growth.'
        ],
        'Banana Insect Pest Disease': [
            'Use insecticides such as imidacloprid or malathion to control pests.',
            'Remove and dispose of infested plant parts.',
            'Implement biological control by introducing natural predators like ladybugs.',
            'Apply neem oil to deter insect pests naturally.'
        ],
        'Banana Moko Disease': [
            'There is no cure for Moko disease. Infected plants should be removed and destroyed.',
            'Control insect vectors by applying insecticides.',
            'Use Moko-resistant banana varieties.',
            'Disinfect tools and equipment to prevent the spread of the bacteria.'
        ],
        'Banana Panama Disease': [
            'Fumigate soil with chemicals like methyl bromide to control the fungus.',
            'Plant resistant banana varieties in affected areas.',
            'Rotate crops to reduce soil-borne pathogens.',
            'Limit the movement of soil and plant debris between fields.'
        ],
        'Banana Yellow Sigatoka Disease': [
            'Use fungicides like tebuconazole or copper-based solutions to control the disease.',
            'Prune and destroy infected leaves to limit spread.',
            'Maintain proper drainage to avoid excessive humidity.',
            'Ensure balanced fertilization to enhance plant resistance.'
        ]
    }

    # Return remedies for the given disease, or an error message if not found
    return jsonify({
        'remedies': remedies.get(disease, ['No remedies found for this disease.']),
    })

# Route to handle contact form submission
@app.route('/send-message', methods=['POST'])
def send_message():
    name = request.form.get('name')
    email = request.form.get('email')
    message_content = request.form.get('message')

    if not name or not email or not message_content:
        return jsonify({'success': False, 'message': 'All fields are required!'}), 400

    try:
        # Create the email message
        msg = Message(
            subject=f"New Contact Form Submission from {name}",
            recipients=['kunalkanse47@gmail.com'],  # Replace with your email
            body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message_content}"
        )

        # Send the email
        mail.send(msg)
        return jsonify({'success': True, 'message': 'Message sent successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)