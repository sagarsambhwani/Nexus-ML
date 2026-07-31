# 📄 Document Classification (NLP Document Categorizer System)

## 📌 System Overview
The **Document Classification System** automatically routes unstructured text documents (resumes, legal contracts, financial audits, HR files, medical records) into pre-defined organizational categories. Automated document categorization streamlines back-office document processing and enterprise content management.

This pipeline utilizes a **Sublinear TF-IDF N-gram Vectorizer** paired with a **Multinomial Naive Bayes Classifier** to assign document categories, compute probability distributions across categories, and extract key domain terms.

---

## 🏗️ Deep-Dive Implementation Architecture

Implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/document_classification/pipeline.py) as a subclass of `BasePipeline`:

```
Raw Unstructured Text -> Sublinear TF-IDF Feature Extractor -> Multinomial Naive Bayes -> Category Class + Probability Distribution + Key Domain Keywords
```

### Class Code Structure & Execution Flow:

```python
class DocumentClassificationPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Document Classification", artifact_name="document_classification_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Synthesizes document corpus ($N=1,500$) across 5 document types:
     - `ENGINEERING_RESUME`: Software engineering vocabulary (Python, Docker, Kubernetes, Microservices, FastAPI, PyTorch).
     - `FINANCE_DOCUMENT`: Fiscal reporting terms (EBITDA, balance sheet, audit, accounts payable, revenue).
     - `HR_RECRUITMENT`: Talent acquisition vocabulary (interviews, onboarding, compensation, employee handbook).
     - `LEGAL_CONTRACT`: Contractual terminology (confidentiality, indemnity, jurisdiction, breach of contract).
     - `HEALTHCARE_RECORD`: Clinical diagnostic vocabulary (patient history, vitals, prescription, physician notes).

2. **NLP Model Pipeline Construction & Training (`train`)**:
   - Assembles `sklearn.pipeline.Pipeline`:
     - `TfidfVectorizer(ngram_range=(1, 2), max_features=3000)`: Extracts unigram and bigram document features.
     - `MultinomialNB(alpha=0.1)`: Trains Naive Bayes with Laplace smoothing ($\alpha = 0.1$).
   - Evaluates multi-class Accuracy (~1.00) and F1-Score (~1.00).
   - Serializes artifact `document_classification_model.joblib`.

3. **Inference & Key Term Extraction (`predict`)**:
   - Vectorizes incoming document text.
   - Computes posterior probability distribution $P(C_k \mid D) \propto P(C_k) \prod P(w_i \mid C_k)$.
   - Identifies highest probability category $C_{\text{best}}$ and confidence score.
   - Extracts top 5 document keyword features driving classification based on non-zero TF-IDF weights.

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why Multinomial Naive Bayes over Deep Learning?
- **Decision**: Selected **Multinomial Naive Bayes (`MultinomialNB`)**.
- **Rationale**:
  - *Extremely Fast $O(N)$ Training & Inference*: Naive Bayes computes log-likelihood additions across word counts in linear time. Ideal for processing large document batches (thousands of pages per minute).
  - *Robustness to Sparse High-Dimensional Vocabularies*: Document text representations contain high-dimensional sparse vector spaces (3,000+ features). Multinomial Naive Bayes handles sparse count vectors natively with zero matrix inversion overhead.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `alpha` | `0.1` | Additive Laplace smoothing parameter preventing zero-probability penalties for novel words unseen in training. |
| `max_features` | `3000` | Focuses feature space on top 3,000 most informative domain terms. |

---

## 📊 Document Categories & Vocabulary Schema

| Document Category | Target Domain | Defining Key Vocabulary Terms |
|---|---|---|
| `ENGINEERING_RESUME` | Technology / IT | Python, Docker, Kubernetes, Microservices, FastAPI, PyTorch, SQL |
| `FINANCE_DOCUMENT` | Accounting / Banking | Balance sheet, EBITDA, cash flow, net margin, audit, equity, tax |
| `HR_RECRUITMENT` | Human Resources | Job description, onboarding, payroll, benefits, appraisal, retention |
| `LEGAL_CONTRACT` | Legal / Compliance | Confidentiality, indemnity, jurisdiction, breach of contract, party |
| `HEALTHCARE_RECORD` | Clinical Medicine | Patient history, diagnosis, prescription, vitals, blood pressure |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **Accuracy**: ~1.00 (100% correct document categorization).
- **F1-Score**: ~1.00 (Weighted multi-class F1 performance).

---

## 🚀 Production Deployment Strategy

1. **OCR Pipeline Integration**: Connect API behind Tesseract OCR / AWS Textract to automatically parse and route incoming PDF scans.
2. **Automated Document Indexing**: Store predicted document category metadata in ElasticSearch / Opensearch for enterprise document search indexing.
