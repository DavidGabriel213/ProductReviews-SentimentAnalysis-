
# 🛒 Nigerian Product Reviews — Sentiment Analyser

A complete NLP project predicting whether Nigerian product reviews are **Positive or Negative** — trained on 10,080 reviews from Jumia, Konga and Jiji platforms in English, Nigerian Pidgin and mixed code-switching language.

## 🌐 Live Demo
**[Try the app →](https://productreviews-sentimentanalysis.onrender.com)**

---

## 📌 Project Overview
This project marks the transition from tabular machine learning to Natural Language Processing — a fundamentally different data type requiring an entirely new preprocessing pipeline. Nigerian e-commerce platforms process thousands of product reviews daily. This system automatically classifies review sentiment, enabling platforms to monitor product quality and seller performance at scale without manual reading.

**What makes this project unique:**
- **Nigerian context** — handles English, Pidgin and code-switching reviews
- **Three-layer stopwords** — English NLTK + Nigerian Pidgin + Domain-specific (~205 words)
- **Logistic Regression wins** — confirming TF-IDF + LR is the classic NLP combination
- **First NLP project** — complete transition from tabular to text data

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 10,080 Nigerian product reviews |
| Platforms | Jumia, Konga, Jiji |
| Languages | English, Nigerian Pidgin, Mixed |
| Target | Sentiment: Positive / Negative |
| Class Balance | Negative: 50.5% / Positive: 49.5% |

### Sample Reviews
```
Positive: "This product dey sweet walahi"
Positive: "Excellent quality, delivery was fast"
Negative: "Abeg avoid this seller e be thief"  
Negative: "Terrible quality waste of my money"
Mixed:    "This product is amazing. Abeg everyone should buy"
```

---

## 🧹 Text Cleaning Pipeline
```python
def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r'<.*?>', '', text)                 # remove HTML tags
    text = re.sub(r'http\S+', '', text)               # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # letters only
    text = text.strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]  # stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # lemmatize
    return ' '.join(tokens)
```

---

## 🛑 Three-Layer Stopword Strategy
| Layer | Words | Example |
|---|---|---|
| English NLTK | ~179 words | "is","the","and","from","of" |
| Nigerian Pidgin | 16 words | "oga","abeg","sha","na","dey","wahala" |
| Domain-specific | 10 words | "product","delivery","seller","jumia" |
| **Total** | **~205 words** | Maximum noise removal for Nigerian reviews |

---

## ⚙️ TF-IDF Vectorizer Configuration
```python
vectorizer = TfidfVectorizer(
    max_features=10000,   # top 10,000 words by score
    ngram_range=(1, 2),   # words AND word pairs ("not good")
    min_df=2,             # word must appear in 2+ reviews
    max_df=0.95,          # remove words in 95%+ of reviews
    sublinear_tf=True     # TF = 1 + log(count) — dampens repetition
)
```

### Why ngram_range=(1,2) matters:
```
Without bigrams:
"not good" → "not" + "good" (positive signal!)

With bigrams:
"not good" → captured as one negative feature ✅
```

---

## 🤖 Model Results
| Model | Accuracy | Negative F1 | Positive F1 | Notes |
|---|---|---|---|---|
| **Logistic Regression** | **100.00%** | **1.00** | **1.00** | **BEST — Deployed ✅** |
| Decision Tree | 87.99% | 0.89 | 0.86 | Single tree limitation |
| Random Forest | 99.95% | 1.00 | 1.00 | Near-perfect |

### Why Logistic Regression wins on TF-IDF:
TF-IDF produces a high-dimensional sparse matrix where word presence has a **linear relationship** with sentiment. "amazing" increases positive probability. "terrible" increases negative. LR excels at this — the same reason it competed closely with RF on the Bank Fraud project.

### Honest Assessment of 100%:
The 100% accuracy reflects clear vocabulary patterns in the synthetic training data. Real-world scraped reviews would score 75-88% due to sarcasm, ambiguous language and mixed sentiment. The honest framing matters more than the number.

---

## 🔄 NLP vs Tabular ML — What Changed
| Stage | Tabular (Projects 5-11) | NLP (This Project) |
|---|---|---|
| Input | Rows and columns | Raw text strings |
| Cleaning | Strip, IQR, fillna | Regex, stopwords, lemmatize |
| Preprocessing | ColumnTransformer | TfidfVectorizer |
| Split rule | Before fit_transform | Before fit_transform ✅ same! |
| Save objects | model + preprocessor | model + vectorizer + label encoder |
| Flask predict | preprocessor.transform(df) | vectorizer.transform([text]) |

---

## 🏗️ Tech Stack
- **Language:** Python
- **NLP:** NLTK (stopwords, lemmatizer)
- **ML:** Scikit-learn (TF-IDF, Logistic Regression)
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3 (Dark Indigo Theme)
- **Deployment:** Render.com
- **Version Control:** GitHub

---

## 📁 Project Structure
```
SentimentAnalysis/
├── models/
│   ├── sentiment_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
├── templates/
│   └── stmnt.html
├── static/
│   └── sendesign.css
├── app.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/SentimentAnalysis
cd SentimentAnalysis
pip install -r requirements.txt
python app.py
```

---

## 💡 Key Learnings
1. **NLP preprocessing** is different from tabular — regex, stopwords, lemmatization replace IQR and OHE
2. **Three-layer stopwords** essential for Nigerian text — NLTK alone misses Pidgin words
3. **TF-IDF is math** — computable by hand; rewards distinctive words, punishes common ones
4. **ngram_range=(1,2)** critical — captures negation patterns like "not good"
5. **LR wins on TF-IDF** — confirming linear relationships between word presence and sentiment
6. **Save vectorizer** — must be identical between training and Flask
7. **100% ≠ perfect** — reflects clean synthetic data; honest documentation matters more
8. **Built in 2 days on Android** — tabular ML foundation transfers directly to NLP

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — 12th project, first NLP. Built on Android phone.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)
