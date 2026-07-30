import pandas as pd
import numpy as np
import re
# NLTK tools
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
df=pd.read_csv("/storage/emulated/0/Download/nigerian_product_reviews_messy.csv")
# Drop dublicates
df=df.drop_duplicates()
#Sentiment
df["Sentiment"] = df["Sentiment"].astype(str).str.strip()
def clean_sentiment(s):
    s = s.strip()
    positive = ["Positive","positive","POSITIVE","Pos","pos","P","1","Good","good","GOOD","👍"]
    negative = ["Negative","negative","NEGATIVE","Neg","neg","N","0","Bad","bad","BAD","👎"]
    if s in positive:
        return "Positive"
    elif s in negative:
        return "Negative"
    else:
        return np.nan
df["Sentiment"] = df["Sentiment"].apply(clean_sentiment)
df = df.dropna(subset=["Sentiment"])
#Rating
df["Rating"] = df["Rating"].astype(str).str.strip()
df["Rating"] = df["Rating"].apply(
    lambda x: x.replace("/5","")
               .replace("stars","")
               .replace("star","")
               .replace("Stars","")
               .strip())
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Rating"] = df["Rating"].clip(1, 5)
df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["Rating"] = df["Rating"].astype(int)
#Platform
df["Platform"] = df["Platform"].astype(str).str.strip()
platform_map = {
    "Jumia": ["Jumia","jumia","JUMIA","Jumia.com","jumia nigeria"],
    "Konga": ["Konga","konga","KONGA","Konga.com","konga ng"],
    "Jiji":  ["Jiji","jiji","JIJI","Jiji.ng","jiji nigeria"],
    }
def clean_platform(p):
    for standard, variants in platform_map.items():
        if p in variants:
            return standard
    return np.nan
df["Platform"] = df["Platform"].apply(clean_platform)
df["Platform"] = df["Platform"].fillna(df["Platform"].mode()[0])
#lematizer
lemmatizer = WordNetLemmatizer()
# English stopwords
stop_words = set(stopwords.words('english'))
# Nigerian Pidgin stopwords
nigerian_stops = {"oga","abeg","sha","na","dey","wahala","sabi","dem","una","wey","abi","joor","ehen","nah","walahi","abii"}
# domain-specific stopwords
domain_stops = {"product","item","order","ordered","delivery","delivered","seller","jumia","konga","jiji","bought","buy"}
#combining all three stopwords
stop_words = stop_words.union(nigerian_stops)
stop_words = stop_words.union(domain_stops)
#text cleaning(reviews)
def clean_text(text):
    text = str(text)
    text = text.lower()
    # HTML tag
    text = re.sub(r'<.*?>', '', text)
    # URLs
    text = re.sub(r'http\S+', '', text)
    # special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # extra whitespace
    text = text.strip()
    tokens = text.split()
    tokens = [w for w in tokens
              if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w)
              for w in tokens]
    return ' '.join(tokens)
df['cleaned_review'] = df['ReviewText'].apply(clean_text)
#Rows where cleaning produced empty string
df = df[df['cleaned_review'].str.strip() != '']
df = df[df['cleaned_review'].str.len() > 5]
df.to_csv("SentimentCleaned.csv",index='false')
