import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class DocumentClassificationPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Document Classification", artifact_name="document_classification_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        doc_types = {
            "ENGINEERING_RESUME": [
                "Software Engineer Python Docker Kubernetes Machine Learning CI CD Microservices React FastAPI SQL PyTorch API REST Data Pipelines",
                "Senior Backend Developer distributed systems Java Spring Boot Cloud AWS Architecture DevOps Git automated testing CI CD",
                "Data Engineer Spark Hadoop SQL Airflow Snowflake ETL Data Warehouse Kafka Python Cloud Architecture Big Data"
            ],
            "FINANCE_DOCUMENT": [
                "Quarterly financial report balance sheet revenue EBITDA cash flow net margin audit investment portfolio asset liability equity tax return",
                "Corporate fiscal budget forecast accounts payable receivable invoice audit risk compliance ledger balance financial audit",
                "Investment banking valuation merger acquisition portfolio return ROI dividend yield asset management hedge fund treasury"
            ],
            "HR_RECRUITMENT": [
                "Job description candidate interview hiring talent acquisition payroll employee benefits onboarding performance review HR policy tenure",
                "Human resources recruitment strategy headcount compensation benefits annual appraisal employee engagement workforce planning retention",
                "Talent sourcing interview scheduling salary negotiation background check offer letter employee handbook HR operations compliance"
            ],
            "LEGAL_CONTRACT": [
                "Non-disclosure agreement confidentiality clause jurisdiction liability indemnity arbitration breach of contract terms and conditions party signature",
                "Master service agreement intellectual property rights indemnification breach termination clause compliance legal counsel governing law",
                "Employment contract non-compete clause copyright trademark patent license covenant enforceable obligation binding agreement"
            ],
            "HEALTHCARE_RECORD": [
                "Patient medical history diagnosis prescription dosage physician clinical note lab results vitals blood pressure cholesterol treatment plan",
                "Hospital admission patient electronic health record EHR radiology MRI scan symptom treatment plan nursing care medical history",
                "Clinical trial protocol physician assessment patient vitals dosage prescription oncology cardiology pathology report laboratory"
            ]
        }

        texts, labels = [], []
        for _ in range(n_samples):
            category = np.random.choice(list(doc_types.keys()))
            sample_text = np.random.choice(doc_types[category])
            words = sample_text.split()
            np.random.shuffle(words)
            texts.append(" ".join(words))
            labels.append(category)

        return pd.DataFrame({"text": texts, "category": labels})

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df["text"]
        y = df["category"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        pipeline = SkPipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
            ("clf", MultinomialNB(alpha=0.1))
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
            return {"category": "UNKNOWN", "confidence": 0.0, "category_distribution": {}}

        pipeline = self.model["pipeline"]
        probs = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_

        best_idx = np.argmax(probs)
        pred_category = str(classes[best_idx])
        confidence = float(probs[best_idx])

        distribution = {str(cls): round(float(p), 4) for cls, p in zip(classes, probs)}

        tfidf = pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        tfidf_vec = tfidf.transform([text]).toarray()[0]
        top_keyword_indices = np.argsort(tfidf_vec)[::-1][:5]
        top_keywords = [str(feature_names[i]) for i in top_keyword_indices if tfidf_vec[i] > 0]

        return {
            "predicted_category": pred_category,
            "confidence": round(confidence, 4),
            "top_keywords_detected": top_keywords,
            "category_distribution": distribution
        }
