#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%%
df = pd.read_csv("/home/abirh/.virtualenvs/FinalProject-Group_AGandJM/Code/cleaned_financial_headlines_finbert2.csv")
X = df["clean_text"]
y = df["label"]
print(df.head())

#%%

# Encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_enc, test_size=0.3, stratify=y_enc, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)



# %%
#Baseline Logistic Regression Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=5
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
logreg.fit(X_train_tfidf, y_train)

# Evaluation on validation set
y_val_pred = logreg.predict(X_val_tfidf)
print("TF-IDF + LogReg Val Accuracy:", accuracy_score(y_val, y_val_pred))
print("TF-IDF + LogReg Val Macro-F1:", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, target_names=le.classes_))


# %%
#evaluation on test set
y_test_pred = logreg.predict(X_test_tfidf)
print("TF-IDF + LogReg Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("TF-IDF + LogReg Test Macro-F1:", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# %%
# Transformer-based model
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long)
        }
        return item

train_dataset = HeadlineDataset(X_train, y_train, tokenizer)
val_dataset   = HeadlineDataset(X_val,   y_val, tokenizer)
test_dataset  = HeadlineDataset(X_test,  y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

#%%
num_labels = len(le.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
model.to(device)
#%%
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss()
#%%
def train_one_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)


from sklearn.metrics import accuracy_score, f1_score

def eval_model(model, data_loader):
    model.eval()
    preds = []
    true  = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds.extend(logits.argmax(dim=1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    macro_f1 = f1_score(true, preds, average="macro")
    return total_loss / len(data_loader), acc, macro_f1
#%%
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, val_f1 = eval_model(model, val_loader)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss:   {val_loss:.4f}")
    print(f"  Val acc:    {val_acc:.4f}")
    print(f"  Val F1(m):  {val_f1:.4f}")
#%%
test_loss, test_acc, test_f1 = eval_model(model, test_loader)
print("TEST loss:", test_loss)
print("TEST acc:", test_acc)
print("TEST macro-F1:", test_f1)
#%%
from sklearn.metrics import classification_report

def get_predictions(model, data_loader):
    model.eval()
    preds = []
    true  = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            preds.extend(logits.argmax(dim=1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    return true, preds

y_test_true, y_test_pred = get_predictions(model, test_loader)
print(classification_report(
    y_test_true, y_test_pred, target_names=le.classes_
))

#%%
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# directory on vm
plots_dir = "/home/abirh/plots"
os.makedirs(plots_dir, exist_ok=True)

#confusion matrix
cm = confusion_matrix(y_test_true, y_test_pred)

# Class names from label encoder
class_names = le.classes_   # ['negative', 'neutral', 'positive']

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix – DistilBERT Sentiment Classifier", fontsize=14)
plt.tight_layout()

# Saving on the VM
out_file = os.path.join(plots_dir, "confusion_matrix_distilbert.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print("Saved confusion matrix to:", out_file)

#%%
cm_norm = confusion_matrix(y_test_true, y_test_pred, normalize='true')

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Purples',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (per class)")
plt.tight_layout()

# Saving on VM
out_file = os.path.join(plots_dir, "normalized_confusion_matrix.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()
print("Saved normalized confusion matrix to:", out_file)
#%%
import pandas as pd
df_test = pd.DataFrame({
    "text": X_test,
    "true_label": le.inverse_transform(y_test_true),
    "pred_label": le.inverse_transform(y_test_pred)
})

errors = df_test[df_test["true_label"] != df_test["pred_label"]]
print(errors.head(10))

#%%
#
import torch
import numpy as np
from torch.nn.functional import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_embeddings(model, data_loader):
    model.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].cpu().numpy()

            # Forward through DistilBERT backbone
            outputs = model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # CLS token embedding = first token from last hidden state
            cls_emb = outputs.last_hidden_state[:, 0, :]
            cls_emb = normalize(cls_emb, p=2, dim=1)  # L2 normalize

            all_embs.append(cls_emb.cpu().numpy())
            all_labels.extend(labels)

    return np.vstack(all_embs), np.array(all_labels)
embs, labels = get_embeddings(model, test_loader)


n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embs)
pca = PCA(n_components=2)
embs_2d = pca.fit_transform(embs)


plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    embs_2d[:, 0],
    embs_2d[:, 1],
    c=labels,
    cmap='viridis',
    alpha=0.7
)

plt.title("PCA Visualization of DistilBERT Embeddings (Colored by True Sentiment)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, ticks=[0,1,2], label="Sentiment (0=neg, 1=neutral, 2=positive)")
plt.tight_layout()

# Save file to VM
out_file = os.path.join(plots_dir, "pca_true_sentiment.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print("Saved PCA true-sentiment plot to:", out_file)

#%%
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    embs_2d[:, 0],
    embs_2d[:, 1],
    c=cluster_labels,
    cmap='Accent',
    alpha=0.7
)

plt.title("KMeans Clusters on DistilBERT Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, ticks=[0,1,2], label="Cluster ID")
plt.tight_layout()

# Save to file on VM
out_file = os.path.join(plots_dir, "pca_kmeans_clusters.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print("Saved KMeans PCA plot to:", out_file)

#%%
##%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

save_dir = "/home/abirh/fin_sentiment_model"
os.makedirs(save_dir, exist_ok=True)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Saved fine-tuned model to:", save_dir)

