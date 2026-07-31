# 💬 Sentiment Analysis (NLP Text Classification System)

## 📌 System Overview
The **Sentiment Analysis System** processes unstructured text feedback (customer product reviews, support tickets, social media posts) to determine underlying emotional tone (`POSITIVE`, `NEUTRAL`, `NEGATIVE`). Automated sentiment analysis enables high-throughput brand monitoring, automated support escalation, and real-time customer sentiment tracking.

This pipeline utilizes a **TF-IDF (Term Frequency-Inverse Document Frequency) Unigram/Bigram Vectorizer** combined with a **Logistic Classifier** to extract semantic signals and score composite sentiment from -1.0 (extremely negative) to +1.0 (extremely positive).

---

## 🏗️ Deep-Dive Implementation Architecture

Implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/sentiment_analysis/pipeline.py) as a subclass of `BasePipeline`:

```mermaid
graph TD
    A[Raw Review Text] --> B[TF-IDF Unigram/Bigram Feature Extraction]
    B[TF-IDF Unigram/Bigram Feature Extraction] --> C[Logistic Classifier]
    C[Logistic Classifier] --> D[Sentiment Category + Confidence Score + Composite Score (-1 to +1)]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
```

### Class Code Structure & Execution Flow:

```python
class SentimentAnalysisPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Sentiment Analysis", artifact_name="sentiment_analysis_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Synthesizes customer review dataset ($N=1,200$): Positive, Neutral, and Negative phrase combinations (e.g., "absolutely amazing product", "average build quality okay", "terrible purchase waste of money").

2. **NLP Pipeline Construction & Training (`train`)**:
   - Assembles `sklearn.pipeline.Pipeline`:
     - `TfidfVectorizer(ngram_range=(1, 2), max_features=2500)`: Extracts unigram and bigram tokens with TF-IDF weighting.
     - `LogisticRegression(random_state=42)`: Fits multi-class logit classifier.
   - Evaluates Accuracy (1.00) and Weighted F1-Score (1.00).
   - Serializes `sentiment_analysis_model.joblib`.

3. **Inference & Composite Sentiment Scoring (`predict`)**:
   - Vectorizes incoming raw text input.
   - Computes class probability distribution $P(\text{POS}), P(\text{NEU}), P(\text{NEG})$.
   - Predicts primary sentiment label and confidence percentage.
   - Derives continuous **Composite Sentiment Score**:
     $$\text{CompositeScore} = P(\text{POSITIVE}) - P(\text{NEGATIVE}) \in [-1.0, +1.0]$$

---


## 💻 API Usage Example

**Endpoint:** `POST /api/v1/predict/sentiment-analysis`

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict/sentiment-analysis' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The product exceeded my expectations! Super fast delivery and great quality."
}'
```

**Expected JSON Response:**
```json
{
  "pipeline": "Sentiment Analysis",
  "status": "success",
  "result": {
    "sentiment": "Positive",
    "score": 0.96
  }
}
```

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why TF-IDF + Logistic Regression over BERT / Transformer LLMs?
- **Decision**: Selected **TF-IDF (Unigram/Bigram) + Logistic Regression**.
- **Rationale**:
  - *Sub-millisecond Latency*: TF-IDF vector multiplication executes in < 2ms on CPU, compared to 150ms+ GPU forward passes for BERT transformers.
  - *Compute Cost*: Enables processing millions of incoming customer reviews daily on single lightweight CPU containers without expensive GPU infrastructure.
  - *N-gram Capture*: Bigram extraction (`ngram_range=(1,2)`) captures key contextual phrases like "not good" or "super fast" effectively.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captures single words and two-word phrase sequences (essential for negation handling). |
| `max_features` | `2500` | Limits vocabulary to the 2,500 most informative n-grams, reducing memory size and filtering out rare typos. |

---

## 📊 Feature Extraction Schema

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$

| Token Sample | Type | Sentiment Weight Contribution |
|---|---|---|
| `"absolutely amazing"` | Bigram | Strong positive weight toward `POSITIVE` |
| `"works great"` | Bigram | Positive weight toward `POSITIVE` |
| `"average build"` | Bigram | Neutral weight toward `NEUTRAL` |
| `"terrible quality"` | Bigram | Strong negative weight toward `NEGATIVE` |
| `"waste of"` | Bigram | Negative weight toward `NEGATIVE` |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **Accuracy**: ~1.00 (100% correct sentiment classification).
- **F1-Score**: ~1.00 (Weighted F1 across all 3 sentiment categories).

---

## 🚀 Production Deployment Strategy

1. **Streaming Analytics Integration**: Integrate API into Kafka / RabbitMQ pipelines to score live customer review feeds.
2. **Alert Escalation**: Automatically route reviews with Composite Score $< -0.70$ directly to urgent customer support queues.
