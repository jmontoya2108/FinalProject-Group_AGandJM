#%%
import pandas as pd
import numpy as np
import nltk
import re
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#%%
#Loading the data

df = pd.read_csv("raw_headlines_data.csv")
df = df[['headline', 'publisher', 'date', 'stock']]
df.dropna(subset=['headline'], inplace=True)

print(df.head())

# %%
#Preprocessing and cleaning data

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)    
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)

df["clean_text"] = df["headline"].astype(str).apply(clean_text)


# %%
# FinBERT sentiment labeling
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

LABELS = ["negative", "neutral", "positive"]

@torch.no_grad()
def finbert_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label_idx = probs.argmax(dim=1).item()
    return LABELS[label_idx]

df["label"] = df["headline"].astype(str).apply(finbert_sentiment)

print(df[["headline", "clean_text", "label"]].head())

# %%
# Check cleanead and label dataset
df.to_csv("cleaned_financial_headlines_finbert2.csv", index=False)

#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=df["label"], palette="viridis")
plt.title("Sentiment Distribution (FinBERT)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

print(df["label"].value_counts())


# %%
top_publishers = df["publisher"].value_counts().head(10).index
subset = df[df["publisher"].isin(top_publishers)]

plt.figure(figsize=(12,6))
sns.countplot(data=subset, x="publisher", hue="label", palette="Set2")
plt.title("Sentiment by Publisher (Top 10)")
plt.xticks(rotation=45)
plt.show()
#%%

top_stocks = df["stock"].value_counts().head(10).index
subset = df[df["stock"].isin(top_stocks)]

plt.figure(figsize=(12,6))
sns.countplot(data=subset, x="stock", hue="label", palette="coolwarm")
plt.title("Sentiment per Stock (Top 10)")
plt.xticks(rotation=45)
plt.show()
# %%
from wordcloud import WordCloud

for sentiment in ["positive", "neutral", "negative"]:
    text = " ".join(df[df["label"] == sentiment]["clean_text"])
    wc = WordCloud(width=1000, height=600, background_color="white").generate(text)
    
    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"WordCloud — {sentiment.capitalize()} Headlines")
    plt.axis("off")
    plt.show()


# %%
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
df["sentiment_score"] = df["label"].map(sentiment_map)

stock_sentiment = df.groupby("stock")["sentiment_score"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
stock_sentiment.plot(kind="bar", color="green")
plt.title("Top 10 least Negative Stocks (Average Sentiment)")
plt.ylabel("Average Sentiment Score")
plt.show()

print(stock_sentiment)


# %%
