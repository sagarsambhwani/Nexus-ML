# Chapter 25: Production Deployment: FastAPI, Docker, Kubernetes, and Cloud Serving

---

## 1. Big Picture

Deploying a model means turning static model weight artifacts into a resilient, high-throughput, low-latency web microservice hosted on cloud infrastructure (AWS, Azure, GCP) or Kubernetes clusters.

This chapter covers FastAPI web endpoints, Docker multi-stage builds, Gunicorn/Uvicorn worker process management, Kubernetes Horizontal Pod Autoscaling (HPA), model compilation (ONNX, TensorRT), and cloud serving.

---

## 2. Intuition

- **Uvicorn / FastAPI**: An asynchronous Python ASGI web server capable of handling thousands of concurrent HTTP connections without blocking.
- **Kubernetes (k8s)**: An automated container orchestrator that monitors microservice health, automatically restarting crashed containers and scaling pod counts from 2 to 50 when CPU/GPU utilization spikes.

---

## 3. Visualization

```text
Kubernetes Autoscaling Production Deployment:

                       Incoming User API Traffic
                                   │
                                   ▼
                    [ Ingress Load Balancer ]
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
  [ Pod 1: FastAPI ]          [ Pod 2: FastAPI ]          [ Pod 3: FastAPI ]
  (Model V2 Container)        (Model V2 Container)        (Model V2 Container)
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   ▼
                  [ Redis Shared Online Feature Store ]
```

---

## 4. Mathematics

### Kubernetes Horizontal Pod Autoscaling (HPA) Formula
Given target CPU utilization $U_{\text{target}}$ (e.g. 70%) and current total metric utilization $U_{\text{current}}$:

$$\text{DesiredPods} = \left\lceil \text{CurrentPods} \times \left( \frac{U_{\text{current}}}{U_{\text{target}}} \right) \right\rceil$$

---

## 5. Python (Production FastAPI Microservice with ONNX Runtime)

```python
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Production ML Inference Service", version="2.0")

class SingleInferenceRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    features: list[float] = Field(..., min_length=4, max_length=4)

class SingleInferenceResponse(BaseModel):
    user_id: int
    prediction_score: float
    latency_ms: float

@app.post("/api/v2/predict", response_model=SingleInferenceResponse)
async def predict_endpoint(request: SingleInferenceRequest):
    import time
    start = time.perf_counter()
    
    if any(np.isnan(request.features)):
        raise HTTPException(status_code=400, detail="NaN values present in feature vector.")
        
    # Simulated Fast Inference Vector Product
    weights = np.array([0.4, -0.1, 0.8, 0.2])
    score = float(1.0 / (1.0 + np.exp(-np.dot(request.features, weights))))
    
    latency = (time.perf_counter() - start) * 1000.0
    return SingleInferenceResponse(user_id=request.user_id, prediction_score=score, latency_ms=latency)
```

---

## 6. Production Kubernetes Deployment YAML Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference-service
  labels:
    app: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
      - name: fastapi-server
        image: nexus-ml/inference-service:v2.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## 7. Under the Hood

- ONNX Runtime compiles model graph nodes into hardware-optimized execution providers (CUDA, TensorRT, OpenVINO, DirectML), fusing activation layers and batch norm layers to cut latency in half.

---

## 8. Engineering Perspective

- **Graceful Shutdown**: Implement SIGTERM handlers in FastAPI microservices to allow active predictions to complete before terminating container pods during rolling updates.

---

## 9. Common Mistakes

1. **Synchronous Blocking IO in FastAPI**: Writing synchronous blocking database calls (`time.sleep()`, heavy file IO) inside async endpoints, blocking the Python event loop.
2. **Missing Container Resource Limits**: Omitting `limits.memory` in Kubernetes manifests, allowing memory leaks in one pod to trigger OOM node crashes for adjacent services.

---

## 10. Interview Questions

### Q1: Compare Batch Inference vs Real-Time Online Inference vs Edge Inference.
**Answer**: Batch inference runs periodically on large historical datasets (e.g. nightly Spark jobs), maximizing throughput at zero real-time latency requirement. Online inference handles real-time HTTP requests, requiring low p99 latency ($< 50\text{ms}$) and high availability. Edge inference runs models locally on client hardware (phones, embedded devices), preserving privacy and working offline without server dependency.

---

## 11. Exercises

1. **Coding**: Export a Scikit-Learn RandomForest model to ONNX format and run inference via `onnxruntime` in Python.
2. **DevOps**: Write a Kubernetes HorizontalPodAutoscaler manifest that scales pods based on CPU utilization > 70%.

---

## 12. Mini Project: Microservice Deployment Package

Build a complete deployment directory containing FastAPI app, ONNX model exporter, Dockerfile, and Kubernetes deployment YAML.

---

## 13. Capstone Integration

Implemented in `api/main.py`, `api/routes.py`, `Dockerfile`, and `docker-compose.yml`.
