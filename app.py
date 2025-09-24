# --- Import Necessary Libraries ---
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model

# --- Initializations ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Download NLTK data (if not already present) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Load Saved Model and Preprocessing Tools ---
# Ensure these files are in the same directory as this script.
try:
    model = load_model('disease_predictor_model.h5')
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure 'disease_predictor_model.h5', 'tfidf_vectorizer.pkl', and 'label_encoder.pkl' are in the same directory.")
    exit()

# --- Text Preprocessing Function (MUST be identical to the one in the training script) ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry for prediction.
    """
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# --- Define the Prediction API Endpoint ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives symptom text from a POST request, processes it,
    and returns the predicted disease as JSON.
    """
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        symptoms_text = data['symptoms']

        # 1. Preprocess the input text
        processed_symptoms = preprocess_text(symptoms_text)

        # 2. Vectorize the preprocessed text using the loaded TF-IDF vectorizer
        vectorized_symptoms = vectorizer.transform([processed_symptoms]).toarray()

        # 3. Predict using the loaded model
        prediction_probabilities = model.predict(vectorized_symptoms)
        
        # 4. Get the index of the highest probability
        predicted_class_index = np.argmax(prediction_probabilities, axis=1)[0]
        
        # 5. Decode the index to get the disease name
        predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]

        # Return the result as JSON
        return jsonify({'predicted_disease': predicted_disease})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


# --- Run the Flask App ---
if __name__ == '__main__':
    print("Starting Flask server... Please wait until the model is loaded.")
    app.run(port=5000, debug=True)