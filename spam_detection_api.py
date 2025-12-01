from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import wget
import os
import uuid

app = Flask(__name__)

############################################
# LOAD DATASET
############################################

# Download dataset once
url = "https://github.com/fahmi54321/spam_detector/raw/refs/heads/main/spam.csv"
if not os.path.exists("spam.csv"):
    wget.download(url, "spam.csv")

# Load data
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Clean columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ["labels", "data"]
df["b_labels"] = df["labels"].map({"ham": 0, "spam": 1})

Y = df["b_labels"].to_numpy()

# Train-test split
df_train, df_test, Ytrain, Ytest = train_test_split(
    df["data"], Y, test_size=0.33
)

# Vectorizer
featurizer = CountVectorizer(decode_error="ignore")
Xtrain = featurizer.fit_transform(df_train)
Xtest = featurizer.transform(df_test)

# Train model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)


############################################
# API: Predict Spam/Ham
############################################

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    X = featurizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return jsonify({
        "text": text,
        "prediction": "spam" if pred == 1 else "ham",
        "spam_probability": float(prob),
        "spam_probability_format": float(f"{prob:.10f}")
    })


############################################
# Run server
############################################

if __name__ == "__main__":
    app.run(debug=True)
