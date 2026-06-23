# Identification of Fake Reviews



A machine learning and deep learning pipeline for detecting fake online reviews (spam reviews) on Amazon and Yelp datasets. This project combines Natural Language Processing (NLP) for text analysis with behavioral analytics for robust classification.

## 🚀 Key Features

* **Hybrid Approach**: Combines textual features (sentiment, readability, embeddings) with behavioral metadata (rating deviation, review counts, etc.).
* **Multiple Architectures**:
  * **Supervised**: XGBoost, LightGBM, Random Forest, SVM, Logistic Regression, etc.
  * **Deep Learning**: ANNs, CNNs, Self-Organizing Maps (SOM).
  * **Unsupervised/Anomaly Detection**: Isolation Forest, One-Class SVM, K-Means.
* **Embeddings**: Utilizes TF-IDF, Word2Vec, GloVe, FastText, and BERT.
* **Live API Deployment**: Includes a high-performance FastAPI web service that dynamically analyzes live inputs!

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Identification-of-fake-reviews.git
   cd Identification-of-fake-reviews
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## 💻 Quick Start & Deployment

**1. Generate and Export the Models**
First, train and save the `.pkl` and `.h5` model files locally:
```bash
python Code/Deploy/train_and_save_models.py
```

**2. Start the Live API Server**
Run the FastAPI web server to accept incoming text over the web:
```bash
uvicorn Code.Deploy.app:app --host 0.0.0.0 --port 8000
```
*Once running, navigate to `http://localhost:8000/docs` in your browser to interact with the models!*

**3. Test the API in Python**
```python
import requests
url = "http://localhost:8000/predict/logistic"
data = {
  "RATING": 4, "VERIFIED_PURCHASE": 1, "REVIEW_TEXT": "amazing product",
  "REVIEW_TITLE": "blast", "AVERAGE_RATING": 3.0, "NUM_REVIEWS": 7
}
print(requests.post(url, json=data).json())
```

## 📊 Results

* **Insight Reports**: Located in `Insight_Reports/` (SageMaker Data Wrangler outputs).
* **Model Evaluation**: Training curves and performance metrics are available in `Results/`.
