from flask import Flask, render_template, request
import joblib
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from scipy.sparse import hstack

app = Flask(__name__)
# model, vectorizer ane labelencoder
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
le = joblib.load('label_encoder.joblib')
#stopwords
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')
nltk.download('punkt_tab')
#lematizer(exact as training)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nigerian_stops = {"oga","abeg","sha","na","dey","wahala","sabi","dem","una","wey","abi","joor","ehen","nah","walahi","abii"}
domain_stops = {"product","item","order","ordered","delivery","delivered","seller","jumia","konga","jiji","bought","buy"}
stop_words = stop_words.union(nigerian_stops)
stop_words = stop_words.union(domain_stops)
# text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    sentiment_class = None
    if request.method == 'POST':
        review_text = request.form['review']
        cleaned = clean_text(review_text)
        X_text = vectorizer.transform([cleaned])
        #preparing rating
        pred_encoded = model.predict(X_text)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
        # class map
        class_map = {
        'Positive': 'result-positive',
        'Negative': 'result-negative'
        }
        sentiment_class = class_map.get(prediction, '')
    return render_template('index.html',prediction = prediction,        sentiment_class = sentiment_class)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    
