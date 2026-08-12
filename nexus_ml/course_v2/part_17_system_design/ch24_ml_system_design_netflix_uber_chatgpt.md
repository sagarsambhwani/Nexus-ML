# Chapter 24: Real-World ML System Design: Netflix, Uber, YouTube, and ChatGPT

---

## 1. Big Picture

In Senior ML Engineering interviews and production architecture meetings, you are tasked with designing large-scale ML systems from scratch. System design requires balancing data ingestion throughput, feature storage, latency SLAs, model evaluation, offline training, and online serving infrastructure.

This chapter covers the end-to-end system design for 5 landmark enterprise platforms:
1. **Netflix Personalized Movie Recommendation**
2. **Uber Dynamic Surge Pricing & ETA Prediction**
3. **YouTube Video Search & Recommendation**
4. **ChatGPT Autonomous Conversational Agent Platform**
5. **Real-Time Financial Fraud Detection Engine**

---

## 2. Intuition & System Design Framework

Every ML System Design answer follows a strict 7-Step Architectural Framework:

```text
 1. Requirements Clarification  ➜ Functional (KPIs) vs Non-Functional (SLA < 50ms, Availability 99.99%)
 2. Metrics & Business Loss      ➜ Offline (ROC-AUC, NDCG@10) vs Online (CTR, Conversion Lift)
 3. Data Ingestion Architecture ➜ Batch (S3/Delta Lake) vs Real-Time Streaming (Kafka/Flink)
 4. Feature Engineering & Store  ➜ Offline Features (Spark/Polars) vs Online Key-Value Store (Redis)
 5. Model Architecture           ➜ Two-Stage Candidate Retrieval + Heavy Ranking Model
 6. Training & Deployment        ➜ Offline Retraining, Shadow Deployment, A/B Testing, Canary
 7. Monitoring & Resilience      ➜ Data Drift (PSI), Fallbacks, Latency Circuit Breakers
```

---

## 3. Visualization: Real-Time Fraud Detection System Architecture

```text
                   REAL-TIME FRAUD DETECTION SYSTEM ARCHITECTURE
                   ─────────────────────────────────────────────

   Client App (Transaction Request)
               │
               ▼
     [ API Gateway / Load Balancer ]
               │
               ├───► [ Kafka Event Stream ] ──► [ Flink Feature Engine ] ──► [ Redis Online Store ]
               │                                                                    │
               ▼                                                                    ▼
     [ FastAPI Fraud Service ] ◄────────────────────────────────────────────────────┘
               │  (Fetches Redis Features in 2ms, Evaluates XGBoost Model in 5ms)
               │
               ├──► Decision: APPROVE (Latency 7ms)
               └──► If Latency Timeout (>30ms) ──► Circuit Breaker Fallback: Rule Engine
```

---

## 4. Mathematics

### System Throughput & Memory Dimensioning
For an API serving $100,000$ Requests Per Second (RPS) where each prediction requires a $512$-dimensional float32 feature vector:

$$\text{Network Bandwidth} = 100,000 \text{ req/sec} \times (512 \times 4 \text{ bytes}) \approx 204.8 \text{ MB/sec}$$

$$\text{Redis QPS} = 100,000 \text{ read ops/sec}$$

To maintain p99 latency under $10\text{ms}$, Redis must be sharded into $K$ cluster nodes:
$$K \ge \frac{\text{Total Read QPS}}{\text{Single Node Max QPS (e.g. 25,000)}} = \frac{100,000}{25,000} = 4 \text{ Sharded Nodes}$$

---

## 5. Python (From Scratch System Design Circuit Breaker & Fallback)

In production ML system design, if a model service experiences high latency spikes, a **Circuit Breaker** automatically redirects traffic to a fast fallback rule engine to preserve system SLAs:

```python
import time

class ProductionMLCircuitBreaker:
    def __init__(self, latency_threshold_sec=0.030, max_consecutive_failures=3):
        self.latency_limit = latency_threshold_sec
        self.max_failures = max_consecutive_failures
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED (Normal ML model), OPEN (Fallback Rules)
        
    def predict(self, feature_vector: list[float]) -> tuple[float, str]:
        if self.state == "OPEN":
            # Fallback Rule Engine (Sub-1ms guarantee)
            return self._fallback_rule_engine(feature_vector), "FALLBACK_RULE_ENGINE"
            
        start = time.perf_counter()
        try:
            score = self._heavy_ml_model_inference(feature_vector)
            elapsed = time.perf_counter() - start
            
            if elapsed > self.latency_limit:
                self._handle_failure()
                
            return score, "HEAVY_ML_MODEL"
            
        except Exception:
            self._handle_failure()
            return self._fallback_rule_engine(feature_vector), "FALLBACK_RULE_ENGINE"

    def _handle_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.state = "OPEN"
            print("CRITICAL: Circuit Breaker Tripped! Routing traffic to Fallback Rule Engine.")

    def _heavy_ml_model_inference(self, vec):
        # Simulated variable latency model
        return float(sum(vec))
        
    def _fallback_rule_engine(self, vec):
        return 1.0 if vec[0] > 5.0 else 0.0

# Test Circuit Breaker
cb = ProductionMLCircuitBreaker(latency_threshold_sec=0.010)
val, src = cb.predict([2.0, 3.0])
print(f"Prediction: {val}, Source: {src}")
```

---

## 6. Real-World Case Studies Overview

### Case 1: Uber Dynamic Surge Pricing
- **Goal**: Predict local supply/demand imbalance every 60 seconds per H3 spatial hex cell.
- **Data Ingestion**: Apache Flink processing real-time driver GPS locations and rider app opens.
- **Feature Store**: Uber Michelangelo Feature Store backed by Redis & Cassandra.
- **Model**: Spatiotemporal GBDT (LightGBM) + ConvLSTM for temporal forecasting.

### Case 2: Netflix Personalization & Page Construction
- **Goal**: Order thousands of movie rows and artwork thumbnails per user.
- **Architecture**: Two-stage candidate generation + Multi-task Neural Network (predicting P(Watch > 30s), P(Like), P(Completion)).
- **Bandit Layer**: Contextual Multi-Armed Bandits dynamically selecting optimal artwork images.

---

## 7. Under the Hood

- **Shadow Deployment**: Deploying a new candidate model version alongside the live production model. Incoming user requests are duplicated to both models, but only the live model's prediction is returned to the user. The candidate model's predictions are logged silently to compare accuracy, latency, and drift in real production conditions before cutover.

---

## 8. Engineering Perspective

- **Graceful Degradation**: Production ML system design must define explicit fallback behaviors for every failure mode (Network timeouts, Redis cache misses, GPU memory OOM).

---

## 9. Common Mistakes

1. **Focusing Only on Algorithms**: Designing a system design solution that spends 45 minutes discussing XGBoost hyper-parameters while completely ignoring data ingestion, Redis caching, latency SLAs, and fallbacks.
2. **Ignoring Data Drift**: Designing static training systems without automated drift detection or retraining triggers.

---

## 10. Interview Questions

### Q1: Design a real-time Fraud Detection System for financial transactions processing 50,000 TPS with a p99 SLA under 20ms.
**Answer**: Structure response into 7 steps: (1) Target <20ms SLA, (2) Streaming ingestion via Kafka, (3) Real-time feature extraction via Flink pushed to Redis online store, (4) In-memory XGBoost model serving compiled via Treelite/ONNX, (5) Shadow deployment + A/B testing, (6) Circuit breaker fallback rules, (7) PSI data drift monitoring.

---

## 11. Exercises

1. **Architecture**: Draw a complete architectural diagram for a YouTube Search & Recommendation System including Candidate Generation, Heavy Ranker, and Diversity Filter.
2. **Coding**: Write a Python microservice that implements Shadow Deployment routing 90% traffic to Model V1 and 10% shadow traffic to Model V2.

---

## 12. Mini Project: End-to-End System Design Spec Document

Write a detailed technical spec document `system_design_chatgpt.md` outlining the architecture, KV-cache cluster sizing, vector store retrieval, and API gateway for a ChatGPT enterprise platform.

---

## 13. Capstone Integration

Maps to system architectures across `api/main.py` and `dashboard/`.
