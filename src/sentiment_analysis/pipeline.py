import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class SentimentAnalysisPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Sentiment Analysis", artifact_name="sentiment_analysis_model.joblib")

    def generate_data(self, n_samples: int = 1200) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        pos_phrases = [
            "absolutely amazing product", "best quality ever", "fast shipping works great",
            "highly recommend to everyone", "fantastic customer support", "exceeded my expectations",
            "super easy to set up and use", "worth every single penny", "love this item so much",
            "outstanding performance and build", "five stars incredible experience", "very satisfied customer"
        ]
        neu_phrases = [
            "received the order today", "item arrived as expected", "average build quality okay",
            "nothing special but works fine", "standard shipping time", "does the basic job",
            "fair price for what it is", "acceptable experience overall", "matches the description online"
        ]
        neg_phrases = [
            "terrible terrible quality broke", "worst purchase waste of money", "horrible customer service",
            "do not buy defective item", "extremely disappointed failed after 1 day", "completely useless junk",
            "misleading description very slow", "return requested absolute scam", "poor packaging damaged"
        ]

        texts, labels = [], []
        for _ in range(n_samples // 3):
            texts.append(np.random.choice(pos_phrases) + " " + np.random.choice(pos_phrases))
            labels.append("POSITIVE")

            texts.append(np.random.choice(neu_phrases) + " " + np.random.choice(neu_phrases))
            labels.append("NEUTRAL")

            texts.append(np.random.choice(neg_phrases) + " " + np.random.choice(neg_phrases))
            labels.append("NEGATIVE")

        return pd.DataFrame({"text": texts, "sentiment": labels})

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df["text"]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        pipeline = SkPipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2500)),
            ("clf", LogisticRegression(random_state=RANDOM_SEED))
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        self.model = {
            "pipeline": pipeline,
            "metrics": {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        text = input_data.get("text", "")
        if not text.strip():
            return {"sentiment": "NEUTRAL", "confidence": 0.50, "sentiment_score": 0.0}

        pipeline = self.model["pipeline"]
        probs = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_

        best_idx = np.argmax(probs)
        pred_sentiment = str(classes[best_idx])
        confidence = float(probs[best_idx])

        class_prob_map = dict(zip(classes, probs))
        composite_score = float(class_prob_map.get("POSITIVE", 0) - class_prob_map.get("NEGATIVE", 0))

        return {
            "text": text,
            "sentiment": pred_sentiment,
            "confidence": round(confidence, 4),
            "composite_sentiment_score": round(composite_score, 4)
        }
