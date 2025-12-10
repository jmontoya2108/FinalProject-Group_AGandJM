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

#%% -------------------------------------
# Loading data
df = pd.read_csv("raw_headlines_data.csv")

#df = df.head(1000)
df = df[['headline', 'publisher', 'date', 'stock']]
df.dropna(subset=['headline'], inplace=True)

df = df.sample(n=10000, random_state=42).reset_index(drop=True)

print(df.head())
#%% 
#Preprocessing

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)      # remove symbols & numbers
    text = re.sub(r"\s+", " ", text).strip()      # collapse spaces

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)

df["clean_text"] = df["headline"].astype(str).apply(clean_text)

#%% -------------------------------------
# FINBERT SENTIMENT + CONFIDENCE
#----------------------------------------
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

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    label_idx = probs.argmax().item()
    confidence = probs[label_idx].item()

    return LABELS[label_idx], confidence

df["label"], df["confidence"] = zip(*df["headline"].astype(str).apply(finbert_sentiment))

print(df[["headline", "clean_text", "label", "confidence"]].head())

#%%
# CORRECTION LAYER

positive_pairs = [
    ("surpass", "estimates"),
    ("beat", "estimates"),
    ("beats", "estimates"),
    ("tops", "estimates"),
    ("above", "expectations"),
    ("better", "expected"),
    ("earnings", "beat"),
    ("revenue", "beat"),
    ("strong", "earnings"),
    ("strong", "revenue")
]

def adjust_sentiment(headline, finbert_label):
    h = headline.lower()


    if finbert_label == "neutral":
        for w1, w2 in positive_pairs:
            if w1 in h and w2 in h:
                return "positive"

    return finbert_label

df["label_adjusted"] = df.apply(
    lambda row: adjust_sentiment(row["headline"], row["label"]),
    axis=1
)


#%% -------------------------------------
# EXPORT DATASET
df.to_csv("cleaned_financial_headlines_10000__random_finbert.csv", index=False)

print("Saved cleaned_financial_headlines_10000_random_finbert.csv")


#%%
import matplotlib.pyplot as plt
import seaborn as sns

pastel_colors = {
    "neutral": "#A8E6A3",
    "positive": "#A3C8FF",
    "negative": "#FFB3B3"
}

hue_order = ["neutral", "positive", "negative"]

plt.figure(figsize=(8,5))
sns.countplot(
    x=df["label"],
    order=hue_order,
    palette=[pastel_colors[h] for h in hue_order]
)

plt.title("Sentiment Distribution (FinBERT)")
plt.xlabel("")
plt.ylabel("")
plt.show()

print(df["label"].value_counts())

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Custom pastel colors
pastel_colors = {
    "neutral": "#7BC67E", 
    "positive": "#6DA8FF", 
    "negative": "#FF8080" 
}

hue_order = ["neutral", "positive", "negative"]

plt.figure(figsize=(8,5))
sns.countplot(
    x=df["label_adjusted"],
    order=hue_order,
    palette=[pastel_colors[h] for h in hue_order]
)

plt.title("Sentiment Distribution (Adjusted Labels)")
plt.xlabel("")
plt.ylabel("")
plt.show()

print(df["label_adjusted"].value_counts())

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

color_map_6 = {
    ("neutral", "FinBERT"):  "#A8E6A3",  
    ("neutral", "Adjusted"): "#7BC67E", 
    
    ("positive", "FinBERT"): "#A3C8FF",  
    ("positive", "Adjusted"): "#6DA8FF", 

    ("negative", "FinBERT"): "#FFB3B3",  
    ("negative", "Adjusted"): "#FF8080"   
}

sentiments = ["neutral", "positive", "negative"]
types = ["FinBERT", "Adjusted"]

# Build comparison table
compare_df = pd.DataFrame({
    "FinBERT": df["label"].value_counts().reindex(sentiments),
    "Adjusted": df["label_adjusted"].value_counts().reindex(sentiments)
})

# X positions
x = np.arange(len(sentiments))
width = 0.35  # bar width

plt.figure(figsize=(11,6))

# Draw bars + add labels
for i, sentiment in enumerate(sentiments):
    for j, label_type in enumerate(types):
        count = compare_df.loc[sentiment, label_type]
        color = color_map_6[(sentiment, label_type)]
        
        # Position left or right within the group
        offset = -width/2 if label_type == "FinBERT" else width/2
        
        # Draw bar
        bar = plt.bar(x[i] + offset, count, width=width, color=color)
        
        # Add number label above the bar
        plt.text(
            x[i] + offset,            
            count + (count * 0.01),   
            f"{int(count)}",           
            ha='center',
            va='bottom',
            fontsize=10
        )

# Labels and aesthetics
plt.xticks(x, sentiments)
plt.title("FinBERT vs Adjusted Sentiment Distribution (6-Color Comparison)")
plt.xlabel("")
plt.ylabel("")
plt.legend(types, title="Label Source")
plt.tight_layout()
plt.show()

print(compare_df)

#%%
#Plot II
publisher_counts = df["publisher"].value_counts()
top_publishers_sorted = publisher_counts.head(10).index.tolist()

subset = df[df["publisher"].isin(top_publishers_sorted)]

subset["publisher"] = pd.Categorical(
    subset["publisher"],
    categories=top_publishers_sorted,
    ordered=True
)


pastel_colors = {
    "neutral": "#A8E6A3",  
    "positive": "#A3C8FF",  
    "negative": "#FFB3B3"   
}

hue_order = ["neutral", "positive", "negative"]

plt.figure(figsize=(12,6))
sns.countplot(
    data=subset,
    x="publisher",
    hue="label",
    palette=[pastel_colors[h] for h in hue_order],
    hue_order=hue_order
)
plt.xlabel("")  
plt.ylabel("")
plt.title("Sentiment by Publisher (Top 10)")
plt.xticks(rotation=45)
plt.show()

#%%
pastel_colors = {
    "neutral": "#A8E6A3",   
    "positive": "#A3C8FF",  
    "negative": "#FFB3B3"
}

hue_order = ["neutral", "positive", "negative"]

top_stocks = df["stock"].value_counts().head(10).index
subset = df[df["stock"].isin(top_stocks)]

plt.figure(figsize=(12,6))
sns.countplot(
    data=subset,
    x="stock",
    hue="label",
    palette=[pastel_colors[h] for h in hue_order],
    hue_order=hue_order
)

plt.title("Sentiment per Stock (Top 10)")
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("")
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
