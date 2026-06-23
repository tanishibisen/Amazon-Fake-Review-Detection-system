import os
import joblib
import numpy as np
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Initialize FastAPI
app = FastAPI(title="Fake Review Detection API", description="API to classify fake vs genuine reviews")

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define input schema
class ReviewData(BaseModel):
    RATING: int
    VERIFIED_PURCHASE: int
    REVIEW_TEXT: str
    REVIEW_TITLE: str
    AVERAGE_RATING: float
    NUM_REVIEWS: int

# Global variables for models
lr_model = None
lr_scaler = None
ann_model = None
ann_scaler = None
tfidf_vectorizer = None

@app.on_event("startup")
def load_models():
    global lr_model, lr_scaler, ann_model, ann_scaler, tfidf_vectorizer
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    try:
        lr_model = joblib.load(os.path.join(models_dir, "logistic_model.pkl"))
        lr_scaler = joblib.load(os.path.join(models_dir, "lr_scaler.pkl"))
        ann_model = tf.keras.models.load_model(os.path.join(models_dir, "ann_model.h5"))
        ann_scaler = joblib.load(os.path.join(models_dir, "ann_scaler.pkl"))
        tfidf_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

def extract_features(data: ReviewData):
    # Stylistic / Simple
    review_length = len(data.REVIEW_TEXT)
    title_length = len(data.REVIEW_TITLE)
    
    # Readability
    readability_fre = flesch_reading_ease(data.REVIEW_TEXT)
    
    # Sentiment & Coherence
    sentiment_score = sid.polarity_scores(data.REVIEW_TEXT)['compound']
    sentiment_cat = 'positive' if sentiment_score > 0.0 else 'negative'
    rating_cat = 'positive' if data.RATING > 3.0 else 'negative'
    coherent_encoded = 1 if sentiment_cat == rating_cat else 0
    
    # Deviation
    rating_deviation = abs(data.RATING - data.AVERAGE_RATING)
    
    # POS Tagging
    doc = nlp(data.REVIEW_TEXT)
    pos_counts = doc.count_by(spacy.attrs.POS)
    num_nouns = pos_counts.get(spacy.parts_of_speech.NOUN, 0)
    num_verbs = pos_counts.get(spacy.parts_of_speech.VERB, 0)
    num_adjectives = pos_counts.get(spacy.parts_of_speech.ADJ, 0)
    num_adverbs = pos_counts.get(spacy.parts_of_speech.ADV, 0)

    # Dictionary representing the extracted numerical/behavioral features
    features = {
        'RATING': data.RATING,
        'VERIFIED_PURCHASE': data.VERIFIED_PURCHASE,
        'REVIEW_LENGTH': review_length,
        'TITLE_LENGTH': title_length,
        'SENTIMENT_SCORE': sentiment_score,
        'COHERENT_ENCODED': coherent_encoded,
        'RATING_DEVIATION': rating_deviation,
        'READABILITY_FRE': readability_fre,
        'NUM_NOUNS': num_nouns,
        'NUM_VERBS': num_verbs,
        'NUM_ADJECTIVES': num_adjectives,
        'NUM_ADVERBS': num_adverbs,
        'AVERAGE_RATING': data.AVERAGE_RATING,
        'NUM_REVIEWS': data.NUM_REVIEWS
    }
    return features

@app.post("/predict/logistic")
def predict_logistic(data: ReviewData):
    if not lr_model or not lr_scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    features = extract_features(data)
    
    # Order required by Logistic Regression (matching training script)
    # ['RATING', 'VERIFIED_PURCHASE', 'REVIEW_LENGTH', 'TITLE_LENGTH', 'SENTIMENT_SCORE', 'COHERENT_ENCODED',
    #  'RATING_DEVIATION', 'READABILITY_FRE', 'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADJECTIVES', 'NUM_ADVERBS',
    #  'AVERAGE_RATING', 'NUM_REVIEWS']
    feature_vector = [
        features['RATING'], features['VERIFIED_PURCHASE'], features['REVIEW_LENGTH'],
        features['TITLE_LENGTH'], features['SENTIMENT_SCORE'], features['COHERENT_ENCODED'],
        features['RATING_DEVIATION'], features['READABILITY_FRE'], features['NUM_NOUNS'],
        features['NUM_VERBS'], features['NUM_ADJECTIVES'], features['NUM_ADVERBS'],
        features['AVERAGE_RATING'], features['NUM_REVIEWS']
    ]
    
    X_scaled = lr_scaler.transform([feature_vector])
    pred = lr_model.predict(X_scaled)[0]
    
    return {"prediction": "Fake" if pred == 1 else "Genuine", "model": "Logistic Regression"}

@app.post("/predict/ann")
def predict_ann(data: ReviewData):
    if not ann_model or not ann_scaler or not tfidf_vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    features = extract_features(data)
    
    # Order required by ANN:
    # ['REVIEW_LENGTH', 'TITLE_LENGTH', 'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADVERBS',
    #  'NUM_ADJECTIVES', 'SENTIMENT_SCORE', 'VERIFIED_PURCHASE', 'RATING', 'RATING_DEVIATION']
    numeric_vector = [
        features['REVIEW_LENGTH'], features['TITLE_LENGTH'], features['NUM_NOUNS'],
        features['NUM_VERBS'], features['NUM_ADVERBS'], features['NUM_ADJECTIVES'],
        features['SENTIMENT_SCORE'], features['VERIFIED_PURCHASE'], features['RATING'],
        features['RATING_DEVIATION']
    ]
    
    X_text = tfidf_vectorizer.transform([data.REVIEW_TEXT]).toarray()
    X_numeric_scaled = ann_scaler.transform([numeric_vector])
    
    X_ann = np.hstack((X_text, X_numeric_scaled))
    pred = ann_model.predict(X_ann)[0][0]
    
    return {"prediction": "Fake" if pred >= 0.5 else "Genuine", "confidence": float(pred), "model": "ANN"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
