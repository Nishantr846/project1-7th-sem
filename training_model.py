# Disease Prediction from Symptoms - Model Training Script
# This script uses the "Symptom2Disease" dataset from Kaggle.
# It demonstrates text preprocessing, model building, training, and saving.

# --- Step 1: Import Necessary Libraries ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Download NLTK data (only needs to be done once) ---
# NLTK (Natural Language Toolkit) is used for text processing.
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Step 2: Load and Inspect the Dataset ---
# IMPORTANT: Before running, upload the 'Symptom2Disease.csv' file to the same
# directory as this script, or provide the correct file path.
try:
    df = pd.read_csv('data/Symptom2Disease.csv')
except FileNotFoundError:
    print("Error: 'Symptom2Disease.csv' not found.")
    print("Please make sure the dataset file is in the correct directory.")
    exit()

print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
df.info()

# --- Step 3: Preprocess the Text Data (NLP Pipeline) ---
# This is a crucial step to convert raw text into a format the model can understand.

# Initialize tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
label_encoder = LabelEncoder()
vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 words

def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry.
    - Converts to lowercase
    - Removes non-alphabetic characters
    - Removes common English 'stopwords' (like 'the', 'a', 'is')
    - Stems words to their root form (e.g., 'running' -> 'run')
    """
    # 1. Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # 2. Tokenize the text (split into words)
    words = text.split()
    # 3. Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    # 4. Join words back into a single string
    return " ".join(words)

print("\n--- Preprocessing Text Data ---")
# Apply the preprocessing function to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete.")
print(df[['text', 'processed_text']].head())


# --- Step 4: Prepare Data for the Model ---

# A. Encode the Disease Labels
# Convert disease names (e.g., 'Hypertension') into numbers (e.g., 0, 1, 2...).
y = label_encoder.fit_transform(df['label'])
# Also, save the mapping from number to disease name for later
disease_names = label_encoder.classes_
num_classes = len(disease_names)
print(f"\nFound {num_classes} unique diseases.")

# B. Vectorize the Symptom Text
# Convert the processed text into numerical vectors using TF-IDF.
# TF-IDF (Term Frequency-Inverse Document Frequency) represents the importance
# of a word in a document relative to a collection of documents (corpus).
X = vectorizer.fit_transform(df['processed_text']).toarray()
print(f"Created a feature matrix of shape: {X.shape}") # (Samples, Features)

# C. Split the Data
# Divide the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% for testing
    random_state=42, # for reproducibility
    stratify=y # Ensures proportional representation of diseases in splits
)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# --- Step 5: Build the Deep Learning Model (MLP) ---

model = Sequential([
    # Input Layer: Dense layer with 'relu' activation.
    # The input_shape must match the number of features from the TF-IDF vectorizer.
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5), # Dropout helps prevent overfitting by randomly setting a fraction of input units to 0.

    # Hidden Layer
    Dense(64, activation='relu'),
    Dropout(0.5),

    # Output Layer: The number of neurons must match the number of diseases.
    # 'softmax' activation is used for multi-class classification to output probabilities.
    Dense(num_classes, activation='softmax')
])

# --- Step 6: Compile the Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use this for integer-encoded labels
    metrics=['accuracy']
)

model.summary()


# --- Step 7: Train the Model ---
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train, y_train,
    epochs=10, # Number of passes through the entire dataset
    batch_size=32, # Number of samples per gradient update
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)
print("--- Model Training Finished ---")


# --- Step 8: Evaluate the Model ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# --- Step 9: Save the Model and Supporting Files ---
# We need to save three things to use this model in our web application:
# 1. The trained model architecture and weights.
# 2. The TF-IDF vectorizer (to process new user input the same way).
# 3. The label encoder (to convert the model's output back to a disease name).

import pickle

# Save the trained model
model.save('disease_predictor_model.h5')

# Save the vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\nModel and preprocessing tools saved successfully!")
print("Files created: disease_predictor_model.h5, tfidf_vectorizer.pkl, label_encoder.pkl")