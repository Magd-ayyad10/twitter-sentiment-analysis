Twitter Sentiment Analysis with Deep Learning (RNN/LSTM) & Streamlit
This project demonstrates a complete Natural Language Processing (NLP) pipeline for sentiment analysis on Twitter data. It includes a Jupyter Notebook for experimenting with and evaluating three different deep learning models (Simple RNN, LSTM, and Bidirectional LSTM) and a user-friendly Streamlit web app to interact with the best-performing model in real-time.
The goal is to classify tweets into four distinct categories: Positive, Negative, Neutral, or Irrelevant.
Features
Four-Class Sentiment Analysis: Classifies text into Positive, Negative, Neutral, or Irrelevant.
Deep Learning Model Comparison: A detailed Jupyter Notebook (twitter_sentiment_analysis.ipynb) that builds, trains, and compares:
Simple RNN
Long Short-Term Memory (LSTM)
Bidirectional LSTM
Interactive Web GUI: A Streamlit app (app.py) that serves the trained Bidirectional LSTM model for real-time predictions.
Full NLP Pipeline: Includes complete text preprocessing steps (cleaning, lowercasing, stop-word removal, lemmatization) and tokenization.
Technology Stack
Python 3.x
Streamlit: For the interactive web application.
TensorFlow / Keras: For building and training the deep learning models.
Pandas: For data loading and manipulation.
NLTK (Natural Language Toolkit): For text preprocessing.
Scikit-learn: For label encoding and train/test splits.
Jupyter Notebook: For model experimentation and analysis.
Project Structure
.
├── app.py                  # The Streamlit web application
├── twitter_sentiment_analysis.ipynb # Jupyter Notebook for model training/evaluation
├── twitter_training.csv    # Training dataset
├── twitter_test.csv        # Test dataset (used in the notebook)
├── requirements.txt        # Python dependencies
└── README.md               # This documentation


How to Use This Project
There are two main parts to this project: the Streamlit App (for predictions) and the Jupyter Notebook (for analysis).
1. Running the Streamlit Web App
This is the interactive part of the project.
Prerequisites:
Python 3.7+
The twitter_training.csv file must be in the same directory as app.py.
Step 1: Clone the Repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name


Step 2: Create a requirements.txt File
Create a new file named requirements.txt and paste the following lines into it. This file lists all the necessary Python libraries.
streamlit
pandas
numpy
nltk
scikit-learn
tensorflow


Step 3: Set Up a Virtual Environment & Install Dependencies
It's highly recommended to use a virtual environment to avoid conflicts with other projects.
# Create a virtual environment
python -m venv venv

# Activate the environment

venv\Scripts\activate


# Install the required libraries
pip install -r requirements.txt


Step 4: Run the Streamlit App
Once the libraries are installed, run the following command in your terminal:
streamlit run app.py


Streamlit will start a local web server and open the app in your default browser.
Note: The very first time you run the app, it will train the model live. This process may take a few minutes. Thanks to Streamlit's caching, this only happens once. All subsequent loads will be fast.
2. Running the Jupyter Notebook
This notebook contains all the code for model development, training, and evaluation.
Prerequisites:
Jupyter Notebook or JupyterLab installed (pip install notebook).
All libraries from requirements.txt installed.
Both twitter_training.csv and twitter_test.csv must be in the same directory.
Step 1: Launch Jupyter
In your terminal (with your virtual environment activated), run:
jupyter notebook


Step 2: Open and Run the Notebook
Your browser will open the Jupyter interface.
Click on twitter_sentiment_analysis.ipynb.
You can run the cells one by one to see the full process, from data loading and cleaning to model training and final comparison.
Model Details
The Jupyter Notebook provides a head-to-head comparison of three recurrent neural network architectures:
Simple RNN: A basic RNN model that serves as a baseline.
LSTM (Long Short-Term Memory): A more advanced RNN capable of learning long-term dependencies, preventing the "vanishing gradient" problem.
Bidirectional LSTM: The best-performing model in this project. It processes the text sequence both forwards and backwards, giving it a much richer understanding of context. This is the model that is deployed in the Streamlit app.

