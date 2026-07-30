import pandas as pd
import numpy as np
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#Loading cleaned data
df=pd.read_csv("/storage/emulated/0/SentimentCleaned.csv")
#Encoding target
le = LabelEncoder()
df['Sentiment_encoded'] = le.fit_transform(df['Sentiment'])
#features
X_text = df['cleaned_review']
X_rating = df[['Rating']]
#target
y = df['Sentiment_encoded']
# Splitting
X_text_train, X_text_test, \
X_rating_train, X_rating_test, \
y_train, y_test = train_test_split(
    X_text, X_rating, y,
    test_size=0.2,
    random_state=42,
    stratify=y)
#Vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True)
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)
from scipy.sparse import hstack
X_rating_train_arr = X_rating_train.values
X_rating_test_arr = X_rating_test.values
X_train_combined = hstack([
    X_text_train_tfidf,
    X_rating_train_arr])
X_test_combined = hstack([
    X_text_test_tfidf,
    X_rating_test_arr])
#Training 
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1)}
results = {}
for name, model in models.items():
    model.fit(X_text_train_tfidf, y_train)
    y_pred = model.predict(X_text_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name}: {acc*100:.2f}%")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))
best_name = max(results, key=results.get)
best_model = models[best_name]
joblib.dump(best_model, '/storage/emulated/0/download/SentimentAnalysisProject/sentiment_model.joblib')
joblib.dump(vectorizer, '/storage/emulated/0/download/SentimentAnalysisProject/tfidf_vectorizer.joblib')
joblib.dump(le, '/storage/emulated/0/download/SentimentAnalysisProject/label_encoder.joblib')
