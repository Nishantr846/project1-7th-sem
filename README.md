# AI Disease Predictor from Symptoms

This project is a web-based application that uses a deep learning model to predict a possible disease based on symptoms provided by the user. It is built with a Python backend using Flask and a simple HTML/JavaScript frontend.

## Project Structure

```
.
├── app.py                  # Flask web server to handle predictions
├── index.html              # Frontend user interface
├── requirements.txt        # Python dependencies
├── training_model.py       # Script to train the prediction model
├── data/
│   └── Symptom2Disease.csv # Dataset for training
└── README.md               # This file
```

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras, Scikit-learn, NLTK
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Dataset**: [Symptom2Disease from Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.7+
- `pip` for package management

### Installation

1. **Clone the repository (or download the project files)**
2. **Create and activate a virtual environment**
   It is highly recommended to use a virtual environment to keep the project's dependencies isolated.

   ```bash
   # Create a virtual environment named 'venv'
   python3 -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```
3. **Install the required packages**
   Install all the necessary Python libraries from the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project

There are two main steps to run this application: training the model and then running the web server.

### Step 1: Train the Model

Before you can run the web application, you need to train the machine learning model. This script will process the data, train the model, and save the necessary files (`disease_predictor_model.h5`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl`).

Run the following command in your terminal:

```bash
python training_model.py
```

This will create the model files in the root directory of the project.

### Step 2: Run the Flask Web Application

Once the model is trained and the files are saved, you can start the Flask server.

```bash
python app.py
```

The server will start, and you should see output similar to this:

```
 * Running on http://127.0.0.1:5000
```

### Step 3: Use the Application

Open your web browser and navigate to the following URL:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

You will see the "AI Disease Predictor" interface. Enter the symptoms in the text box and click the "Predict Disease" button to get a prediction.

## Disclaimer

This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
