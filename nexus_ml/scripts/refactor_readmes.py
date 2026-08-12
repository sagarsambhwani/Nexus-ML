import os
import re
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

# The examples from api/schemas.py and endpoints
PIPELINES = {
    "fraud_detection": {
        "endpoint": "fraud-detection",
        "name": "Fraud Detection",
        "payload": {
            "amount": 250.50,
            "time_hour": 2,
            "velocity_1h": 4,
            "location_risk": 0.85,
            "v1": 1.8,
            "v2": -0.5,
            "v3": 0.2,
            "v4": 1.1,
            "v5": -0.8
        },
        "response": {
            "pipeline": "Fraud Detection",
            "status": "success",
            "result": {
                "fraud_probability": 0.92,
                "risk_tier": "HIGH_RISK",
                "top_factors": {
                  "location_risk": 0.45,
                  "velocity_1h": 0.32,
                  "amount": 0.15
                }
            }
        }
    },
    "credit_risk": {
        "endpoint": "credit-risk",
        "name": "Credit Risk Prediction",
        "payload": {
            "credit_score": 640,
            "annual_income": 55000,
            "dti_ratio": 0.42,
            "loan_amount": 25000,
            "delinquencies_2yr": 1,
            "employment_years": 3
        },
        "response": {
            "pipeline": "Credit Risk Prediction",
            "status": "success",
            "result": {
                "default_probability": 0.12,
                "risk_tier": "AA (Near Prime)",
                "decision": "APPROVED"
            }
        }
    },
    "customer_churn": {
        "endpoint": "customer-churn",
        "name": "Customer Churn Prediction",
        "payload": {
            "tenure": 6,
            "monthly_charges": 89.90,
            "total_charges": 539.40,
            "contract_type": 0,
            "support_tickets": 4,
            "paperless_billing": 1
        },
        "response": {
            "pipeline": "Customer Churn Prediction",
            "status": "success",
            "result": {
                "churn_probability": 0.85,
                "retention_action": "High Risk - Offer Discount"
            }
        }
    },
    "house_prices": {
        "endpoint": "house-prices",
        "name": "House Price Prediction",
        "payload": {
            "sqft": 2200,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "location_score": 8.5,
            "house_age": 10,
            "garage_cars": 2,
            "dist_city_km": 6.2
        },
        "response": {
            "pipeline": "House Price Prediction",
            "status": "success",
            "result": {
                "predicted_price": 425000,
                "price_lower_bound": 410000,
                "price_upper_bound": 440000
            }
        }
    },
    "recommendation": {
        "endpoint": "recommendation",
        "name": "Recommendation System",
        "payload": {
            "user_id": "USER_005",
            "category": "Electronics",
            "top_n": 5
        },
        "response": {
            "pipeline": "Recommendation System",
            "status": "success",
            "result": {
                "recommendations": [
                    {"item_id": "ITEM_102", "match_score": 0.95},
                    {"item_id": "ITEM_405", "match_score": 0.88}
                ]
            }
        }
    },
    "demand_forecasting": {
        "endpoint": "demand-forecasting",
        "name": "Demand Forecasting",
        "payload": {
            "store_id": "STORE_101",
            "horizon_days": 7,
            "is_promo": 1
        },
        "response": {
            "pipeline": "Demand Forecasting",
            "status": "success",
            "result": {
                "forecasts": [120, 135, 142, 110, 95, 205, 210]
            }
        }
    },
    "predictive_maintenance": {
        "endpoint": "predictive-maintenance",
        "name": "Predictive Maintenance",
        "payload": {
            "vibration_hz": 68.5,
            "temperature_c": 92.3,
            "pressure_psi": 78.0,
            "rpm": 2800,
            "sensor_noise_std": 3.2,
            "operating_hours": 6500
        },
        "response": {
            "pipeline": "Predictive Maintenance",
            "status": "success",
            "result": {
                "failure_probability": 0.78,
                "rul_hours": 45,
                "status": "CRITICAL_WARNING"
            }
        }
    },
    "medical_diagnosis": {
        "endpoint": "medical-diagnosis",
        "name": "Medical Diagnosis Support",
        "payload": {
            "age": 54,
            "glucose": 145.0,
            "blood_pressure": 95.0,
            "bmi": 32.4,
            "hba1c": 6.8,
            "family_history": 1,
            "smoker": 0
        },
        "response": {
            "pipeline": "Medical Diagnosis Support",
            "status": "success",
            "result": {
                "disease_risk": 0.82,
                "guidance": "Elevated HbA1c and Glucose. Consult physician."
            }
        }
    },
    "sentiment_analysis": {
        "endpoint": "sentiment-analysis",
        "name": "Sentiment Analysis",
        "payload": {
            "text": "The product exceeded my expectations! Super fast delivery and great quality."
        },
        "response": {
            "pipeline": "Sentiment Analysis",
            "status": "success",
            "result": {
                "sentiment": "Positive",
                "score": 0.96
            }
        }
    },
    "document_classification": {
        "endpoint": "document-classification",
        "name": "Document Classification",
        "payload": {
            "text": "Senior Software Engineer with 6 years experience in Python, Docker, Kubernetes, microservices, and FastAPI backend development."
        },
        "response": {
            "pipeline": "Document Classification",
            "status": "success",
            "result": {
                "category": "Technology/Engineering",
                "confidence": 0.94,
                "keywords": ["Python", "Docker", "FastAPI"]
            }
        }
    },
    "defect_detection": {
        "endpoint": "defect-detection",
        "name": "Defect Detection",
        "payload": {
            "mean_intensity": 110.0,
            "std_intensity": 35.2,
            "edge_pixel_density": 0.22,
            "contrast_ratio": 5.1,
            "surface_roughness": 4.2,
            "anomaly_patch_max": 0.82
        },
        "response": {
            "pipeline": "Defect Detection",
            "status": "success",
            "result": {
                "defect_type": "Surface Scratch",
                "severity": "High",
                "qc_status": "FAIL"
            }
        }
    },
    "customer_segmentation": {
        "endpoint": "customer-segmentation",
        "name": "Customer Segmentation",
        "payload": {
            "annual_income_k": 105.0,
            "spending_score": 88.0,
            "frequency_purchases": 22.0,
            "recency_days": 14.0
        },
        "response": {
            "pipeline": "Customer Segmentation",
            "status": "success",
            "result": {
                "cluster_id": 2,
                "persona": "High-Value Loyal",
                "strategy": "VIP Rewards Program"
            }
        }
    }
}

def generate_mermaid(flow_string):
    parts = [p.strip() for p in flow_string.split("->")]
    lines = ["```mermaid", "graph TD"]
    
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(len(parts) - 1):
        lines.append(f"    {chars[i]}[{parts[i]}] --> {chars[i+1]}[{parts[i+1]}]")
    
    lines.append(f"    style A fill:#f9f,stroke:#333,stroke-width:2px")
    lines.append(f"    style {chars[len(parts)-1]} fill:#bbf,stroke:#333,stroke-width:2px")
    lines.append("```")
    return "\n".join(lines)

for p_key, meta in PIPELINES.items():
    readme_path = os.path.join(SRC_DIR, p_key, "README.md")
    if not os.path.exists(readme_path):
        continue
        
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if "API Usage Example" in content:
        continue # already done
        
    # 1. Replace Mermaid
    # Look for a line containing " -> "
    lines = content.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if " -> " in line and line.startswith("```"):
            pass # skip
        elif " -> " in line and not line.startswith(" "):
            # This is the flow line
            mermaid = generate_mermaid(line)
            # Find the enclosing ``` and remove them
            if len(new_lines) > 0 and new_lines[-1] == "```":
                new_lines.pop()
            new_lines.append(mermaid)
            i += 1
            if i < len(lines) and lines[i] == "```":
                i += 1
            continue
        new_lines.append(line)
        i += 1
        
    new_content = "\n".join(new_lines)
    
    # 2. Inject API Usage Example
    api_section = f"""
## 💻 API Usage Example

**Endpoint:** `POST /api/v1/predict/{meta["endpoint"]}`

```bash
curl -X 'POST' \\
  'http://localhost:8000/api/v1/predict/{meta["endpoint"]}' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{json.dumps(meta["payload"], indent=2)}'
```

**Expected JSON Response:**
```json
{json.dumps(meta["response"], indent=2)}
```

---
"""
    new_content = new_content.replace("## 🎯 Engineering Decision", api_section + "\n## 🎯 Engineering Decision")
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
print("Updated all READMEs!")
