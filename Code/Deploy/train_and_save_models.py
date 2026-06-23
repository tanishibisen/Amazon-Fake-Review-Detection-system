import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_and_save():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading dataset...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "../../Datasets/amazon_reviews_training.csv")
    df = pd.read_csv(dataset_path)
    df = df.dropna()

    # ---------------------------------------------------------
    # 1. Train and Save Logistic Regression (Numerical Features)
    # ---------------------------------------------------------
    print("Training Logistic Regression...")
    lr_features = ['RATING', 'VERIFIED_PURCHASE', 'REVIEW_LENGTH',
                'TITLE_LENGTH', 'SENTIMENT_SCORE', 'COHERENT_ENCODED',
                'RATING_DEVIATION', 'READABILITY_FRE', 'NUM_NOUNS',
                'NUM_VERBS', 'NUM_ADJECTIVES', 'NUM_ADVERBS',
                'AVERAGE_RATING', 'NUM_REVIEWS']
    
    X_lr = df[lr_features]
    Y = df['LABEL_ENCODED']

    # Scale features
    lr_scaler = MinMaxScaler()
    X_lr_scaled = lr_scaler.fit_transform(X_lr)

    # Train model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_lr_scaled, Y)

    # Save
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_model.pkl"))
    joblib.dump(lr_scaler, os.path.join(models_dir, "lr_scaler.pkl"))
    print("Logistic Regression saved!")

    # ---------------------------------------------------------
    # 2. Train and Save Keras ANN (Text + Numerical Features)
    # ---------------------------------------------------------
    print("Training Keras ANN...")
    features_text = df['REVIEW_TEXT'].values
    ann_features_numeric = ['REVIEW_LENGTH', 'TITLE_LENGTH',
                      'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADVERBS',
                       'NUM_ADJECTIVES', 'SENTIMENT_SCORE',
                       'VERIFIED_PURCHASE', 'RATING', 'RATING_DEVIATION']
    
    X_ann_numeric = df[ann_features_numeric].values

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000) # limiting to 5000 to keep model size reasonable for deployment
    X_text_tfidf = tfidf.fit_transform(features_text).toarray()

    # Standardize numeric
    ann_scaler = StandardScaler()
    X_ann_numeric_scaled = ann_scaler.fit_transform(X_ann_numeric)

    # Combine
    X_ann = np.hstack((X_text_tfidf, X_ann_numeric_scaled))

    # Build ANN
    ann_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_ann.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ann_model.fit(X_ann, Y.values, epochs=10, batch_size=32, verbose=1) # using 10 epochs for faster execution right now

    # Save
    ann_model.save(os.path.join(models_dir, "ann_model.h5"))
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(ann_scaler, os.path.join(models_dir, "ann_scaler.pkl"))
    print("ANN Model and preprocessors saved!")

if __name__ == "__main__":
    train_and_save()
