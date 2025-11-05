import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping

# --- Download NLTK data (runs once) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    return True

download_nltk_data()

# --- Text Preprocessing Function (from your notebook) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = []
    for word in text.split():
        if word not in stop_words:
            tokens.append(lemmatizer.lemmatize(word))
    return ' '.join(tokens)

# --- Model & Preprocessing Parameters (from your notebook) ---
VOCAB_SIZE = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 64
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = '<OOV>'

# --- Model Training Function (Cached) ---
@st.cache_resource
def load_and_train_model():
    # 1. Load Data
    col_names = ['TweetID', 'Entity', 'Sentiment', 'Content']
    try:
        df_train = pd.read_csv('twitter_training.csv', names=col_names, header=None)
    except FileNotFoundError:
        st.error("ERROR: 'twitter_training.csv' not found. Please place it in the same folder as app.py.")
        return None, None, None
        
    df_train.dropna(subset=['Content', 'Sentiment'], inplace=True)
    
    # 2. Clean Text
    df_train['Clean_Content'] = df_train['Content'].apply(clean_text)
    
    # 3. Label Encoding
    encoder = LabelEncoder()
    df_train['Sentiment_Encoded'] = encoder.fit_transform(df_train['Sentiment'])
    y_train_full = df_train['Sentiment_Encoded'].values
    num_classes = len(encoder.classes_)
    
    # 4. Tokenization & Padding
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(df_train['Clean_Content'])
    
    X_train_full_seq = tokenizer.texts_to_sequences(df_train['Clean_Content'])
    X_train_full_pad = pad_sequences(X_train_full_seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
    # 5. Create Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_pad, 
        y_train_full, 
        test_size=0.1, 
        random_state=42, 
        stratify=y_train_full
    )
    
    # 6. Define Bidirectional LSTM Model (Model 3 from notebook)
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    # 7. Train Model
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=10, # Kept it to 10 for faster first-load, notebook was 20 with early stopping
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    
    return model, tokenizer, encoder

# --- Streamlit App UI ---

st.title("🐦 Twitter Sentiment Analyzer")
st.markdown("""
This app uses a Bidirectional LSTM model (trained on the provided dataset) 
to classify the sentiment of a tweet.
""")

# Load model (with caching)
with st.spinner("Loading and training the model... This may take a few minutes on first run."):
    model, tokenizer, encoder = load_and_train_model()

if model:
    st.success("Model is trained and ready!")

    # User Input
    user_tweet = st.text_area("Enter your tweet to analyze:", "This is the best game I have ever played!")

    if st.button("Analyze Sentiment"):
        if user_tweet:
            # 1. Clean the input
            cleaned_tweet = clean_text(user_tweet)
            
            # 2. Tokenize and pad
            seq = tokenizer.texts_to_sequences([cleaned_tweet])
            padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
            
            # 3. Predict
            prediction_probs = model.predict(padded_seq)
            prediction = np.argmax(prediction_probs, axis=1)
            
            # 4. Decode label
            predicted_label = encoder.inverse_transform(prediction)[0]
            
            # 5. Display result
            st.subheader("Analysis Result")
            
            if predicted_label == "Positive":
                st.markdown(f"## <div style='color:green;display:inline-block;'>{predicted_label} 😊</div>", unsafe_allow_html=True)
            elif predicted_label == "Negative":
                st.markdown(f"## <div style='color:red;display:inline-block;'>{predicted_label} 😠</div>", unsafe_allow_html=True)
            elif predicted_label == "Neutral":
                st.markdown(f"## <div style='color:blue;display:inline-block;'>{predicted_label} 😐</div>", unsafe_allow_html=True)
            else: # Irrelevant
                st.markdown(f"## <div style='color:gray;display:inline-block;'>{predicted_label} 🤷</div>", unsafe_allow_html=True)

            st.write("Confidence Scores:")
            confidence_df = pd.DataFrame(prediction_probs, columns=encoder.classes_)
            st.bar_chart(confidence_df.T)
            
        else:
            st.warning("Please enter a tweet to analyze.")
