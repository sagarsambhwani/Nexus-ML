# AI/ML Backend Engineer Interview Guide

## 1. Interview Structure

A strong interview can be divided into **7 rounds/areas**:

| Area                                  | Suggested Time | What to Evaluate                            |
| ------------------------------------- | -------------: | ------------------------------------------- |
| Python / Backend Fundamentals         |      30–40 min | Coding, Python internals, APIs, concurrency |
| Machine Learning Theory               |      25–30 min | Fundamentals, intuition, trade-offs         |
| Deep Learning + Transformers          |      30–40 min | Architecture and training understanding     |
| FastAPI + Async + Multithreading      |      25–30 min | Production backend knowledge                |
| LLMs + RAG + Fine-tuning + Evaluation |      40–50 min | Applied GenAI expertise                     |
| Multimodality + Voice Agents          |      25–30 min | Modern AI systems understanding             |
| RAG Chatbot System Design + Scaling   |      45–60 min | Architecture, scalability, reliability      |

---

# 2. Python / Backend Fundamentals

## Core Python Questions

### Q1. What is the difference between a list, tuple, set, and dictionary?

**Expected answer:**

* `list`: ordered, mutable, allows duplicates.
* `tuple`: ordered, immutable, allows duplicates.
* `set`: unordered collection of unique elements.
* `dict`: key-value mapping, keys must be hashable.
* Average lookup in a set/dictionary is approximately **O(1)**.

### Q2. What is the difference between `is` and `==`?

`==` checks value equality.

`is` checks object identity.

```python
a = [1, 2]
b = [1, 2]

a == b  # True
a is b  # False
```

### Q3. Explain shallow copy vs deep copy.

```python
import copy

b = copy.copy(a)
c = copy.deepcopy(a)
```

A shallow copy copies the outer object but can retain references to nested objects.

A deep copy recursively copies nested objects.

### Q4. What are decorators?

A decorator wraps a function/class to modify or extend its behavior without changing its source code.

```python
def logger(func):
    def wrapper(*args, **kwargs):
        print("calling")
        result = func(*args, **kwargs)
        print("done")
        return result
    return wrapper
```

### Q5. What are generators?

Generators produce values lazily using `yield`.

They are useful for:

* Large files
* Streaming data
* Memory-efficient pipelines
* Infinite sequences

```python
def numbers():
    for i in range(10):
        yield i
```

### Q6. What is the GIL?

The **Global Interpreter Lock** in CPython allows only one thread to execute Python bytecode at a time within a process.

Important nuance:

* Threads can still be useful for I/O-bound work.
* CPU-bound Python code generally benefits more from multiprocessing or native/vectorized implementations.
* The GIL does not mean Python cannot perform concurrent I/O.

---

## Coding Questions

### Easy

1. Reverse a string.
2. Find duplicate elements in an array.
3. Find the first non-repeating character.
4. Count word frequencies.
5. Check whether two strings are anagrams.
6. Merge two sorted arrays.
7. Find the second-largest number.
8. Implement a stack using a list.
9. Implement a queue.
10. Find whether a linked list contains a cycle.

### Medium

1. Implement an LRU cache.
2. Implement a rate limiter.
3. Merge overlapping intervals.
4. Find top-K frequent elements.
5. Implement a producer-consumer queue.
6. Implement retry with exponential backoff.
7. Implement a TTL cache.
8. Implement a simple task scheduler.

### Backend-oriented coding problem

**Build a function that processes 1,000 URLs with a maximum concurrency of 20.**

Look for:

* Async I/O
* Concurrency limiting
* Exception handling
* Timeout handling
* Retry logic
* Result aggregation

---

# 3. Machine Learning Theory

## Fundamentals

### Q1. Supervised vs unsupervised vs reinforcement learning?

**Supervised:** learns from labeled `(X, y)` data.

**Unsupervised:** discovers structure without labels.

**Reinforcement learning:** learns actions based on rewards from an environment.

---

### Q2. Explain bias vs variance.

**High bias:**

* Model too simple
* Underfitting

**High variance:**

* Model too sensitive to training data
* Overfitting

Ask the candidate:

> How would you reduce overfitting?

Expected possibilities:

* More data
* Regularization
* Data augmentation
* Dropout
* Early stopping
* Simpler model
* Cross-validation

---

### Q3. Why do we split data into train/validation/test?

* **Train:** learn parameters.
* **Validation:** tune hyperparameters/model choices.
* **Test:** estimate final generalization performance.

Follow-up:

> What is data leakage?

Expected answer: information from outside the training process improperly influences training/model selection.

---

### Q4. Precision vs recall?

**Precision**

`TP / (TP + FP)`

Of the things predicted positive, how many were actually positive?

**Recall**

`TP / (TP + FN)`

Of all actual positives, how many did we identify?

Ask:

> For a medical screening system, which might be more important?

Usually recall, depending on the application.

---

### Q5. What is cross-validation?

A technique where training data is divided into multiple folds, with each fold serving as validation in turn.

Useful when:

* Dataset is relatively small.
* We want a more reliable estimate of model performance.

---

### Q6. Explain regularization.

Regularization discourages overly complex models.

Examples:

* L1
* L2
* Dropout
* Weight decay
* Early stopping

---

### Q7. What happens when the learning rate is too high or too low?

**Too high:**

* Training may diverge.
* Loss can oscillate.

**Too low:**

* Training becomes very slow.
* May get stuck around poor regions.

---

# 4. Deep Learning

## Neural Networks

### Q1. What is backpropagation?

Backpropagation computes gradients of the loss with respect to model parameters using the chain rule.

Then an optimizer updates the parameters.

Typical loop:

**Forward pass → Loss → Backward pass → Optimizer step**

---

### Q2. Why do we need activation functions?

Without nonlinear activations, stacking linear layers still produces essentially one linear transformation.

Common activations:

* ReLU
* GELU
* Sigmoid
* Tanh
* SiLU/Swish

---

### Q3. Why is ReLU commonly used?

It is computationally simple and generally helps mitigate vanishing gradients compared with sigmoid/tanh in deep networks.

Potential issue:

**Dying ReLU** — neurons can become stuck outputting zero.

---

# 5. Transformers

## Must-Know Questions

### Q1. Explain self-attention.

Given query, key, and value matrices:

`Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V`

The model determines how strongly each token should attend to other tokens.

---

### Q2. Why divide by √dₖ?

As dimensionality increases, dot products can become large.

Large values can push softmax into saturated regions, producing poor gradients.

Scaling stabilizes training.

---

### Q3. What is multi-head attention?

Instead of performing one attention operation, the model performs multiple attention operations in parallel.

Different heads can learn different relationships.

---

### Q4. Why do Transformers need positional information?

Self-attention itself does not inherently encode token order.

Position information allows the model to distinguish:

> "dog bites man"

from

> "man bites dog"

Approaches include:

* Sinusoidal positional embeddings
* Learned positional embeddings
* RoPE
* ALiBi

---

### Q5. Encoder vs decoder architecture?

**Encoder:**

* Bidirectional context
* Common in representation/understanding models

**Decoder:**

* Causal/autoregressive attention
* Predicts future tokens from previous tokens

Examples:

* BERT → encoder
* GPT-style models → decoder
* T5 → encoder-decoder

---

### Q6. Why is causal masking required?

During autoregressive generation, token `t` should not see future tokens.

A causal mask prevents attention to future positions.

---

### Q7. What is KV caching?

During autoregressive generation, previously computed keys and values can be cached instead of recomputed for every new token.

This significantly improves generation latency.

Follow-up:

> What is the cost?

KV cache consumes substantial GPU memory, especially for:

* Long contexts
* Large batch sizes
* Large models

---

# 6. FastAPI + Async + Multithreading

## FastAPI

### Q1. Why use FastAPI?

Expected points:

* ASGI-based
* Async support
* Type hints
* Pydantic validation
* Automatic OpenAPI documentation
* Good performance
* Dependency injection

---

### Q2. What is ASGI?

ASGI is an asynchronous server interface for Python applications.

It supports applications involving:

* HTTP
* WebSockets
* Long-lived connections
* Async workloads

---

### Q3. `async def` vs `def` endpoint?

An async endpoint is appropriate when the workload performs asynchronous I/O.

For example:

```python
@app.get("/users")
async def get_users():
    result = await db.fetch(...)
    return result
```

Using async does **not** automatically make CPU-heavy work faster.

---

### Q4. What blocks an async event loop?

Examples:

```python
time.sleep(10)
```

CPU-heavy operations:

```python
for i in range(10**10):
    ...
```

Blocking synchronous database/network clients.

---

### Q5. How would you handle CPU-heavy work in FastAPI?

Possible solutions:

* Worker processes
* Background task queues
* ProcessPoolExecutor
* Dedicated inference workers
* External services

Do not simply put heavy CPU work directly into the event loop.

---

## Async vs Threads vs Processes

| Workload          | Preferred Approach    |
| ----------------- | --------------------- |
| Async HTTP calls  | Async I/O             |
| Database I/O      | Async client          |
| File/network I/O  | Async or threads      |
| CPU-heavy Python  | Multiprocessing       |
| ML inference      | Dedicated workers/GPU |
| Long-running jobs | Task queue            |

---

# 7. LLM Fundamentals

## Questions

### Q1. What happens when you call an LLM?

High-level pipeline:

**Prompt → Tokenization → Embeddings → Transformer layers → Logits → Sampling/decoding → Tokens → Text**

---

### Q2. Temperature vs top-p?

**Temperature** modifies the sharpness of the probability distribution.

Higher temperature → more randomness.

Lower temperature → more deterministic.

**Top-p** restricts sampling to the smallest group of tokens whose cumulative probability exceeds `p`.

---

### Q3. What is hallucination?

An LLM generates information that appears plausible but is unsupported or incorrect.

Causes can include:

* Missing knowledge
* Ambiguous prompts
* Poor retrieval
* Distribution mismatch
* Sampling
* Model limitations

---

# 8. RAG

## Core RAG Architecture

**User Query**

↓

**Query Processing**

↓

**Embedding Model**

↓

**Vector Search**

↓

**Top-K Documents**

↓

**Reranking / Filtering**

↓

**Context Construction**

↓

**LLM**

↓

**Answer + Citations**

---

## Q1. Why use RAG instead of fine-tuning?

RAG is useful when:

* Knowledge changes frequently.
* Documents are private.
* We need citations.
* We need controllable retrieval.
* We don't want to retrain the model for every document update.

Fine-tuning is more useful for:

* Behavior
* Style
* Task adaptation
* Domain-specific patterns
* Structured output behavior

---

## Q2. How do you choose chunk size?

There is no universal optimal size.

Consider:

* Document structure
* Retrieval granularity
* Embedding model
* Context window
* Query type
* Need for surrounding context

Test multiple chunking strategies empirically.

---

## Q3. What is hybrid search?

Combines semantic/vector search with lexical search such as BM25.

Useful because:

* Vector search captures semantic similarity.
* Keyword search captures exact terms, IDs, names, codes, etc.

---

## Q4. Why use a reranker?

Initial retrieval may return several approximately relevant documents.

A reranker performs a more expensive relevance assessment and reorders candidates.

Typical pipeline:

**Retrieve 50–100 → Rerank → Keep top 5–10**

---

# 9. RAG Without and With OCR

## Text PDFs

Pipeline:

**PDF → Text extraction → Cleaning → Chunking → Embeddings → Vector DB**

## Scanned PDFs

Pipeline:

**PDF → Page images → OCR → Layout/structure extraction → Cleaning → Chunking → Embeddings**

Potential OCR problems:

* Tables
* Columns
* Headers/footers
* Handwriting
* Mathematical notation
* Low-quality scans

---

## Strong Follow-up

Ask:

> How would you make a RAG system understand a 100-page financial PDF containing tables, charts, and scanned pages?

A strong candidate should discuss:

* OCR
* Layout detection
* Table extraction
* Metadata
* Page-level references
* Multimodal models
* Chunking by semantic structure
* Hybrid retrieval
* Reranking
* Citations
* Evaluation

---

# 10. Fine-tuning

## Q1. Fine-tuning vs LoRA vs prompt engineering?

### Prompt Engineering

Change the input.

Lowest operational complexity.

### Full Fine-tuning

Update most/all model parameters.

More expensive and resource intensive.

### LoRA / PEFT

Train small adapter matrices while keeping most base-model parameters frozen.

Benefits:

* Lower GPU memory requirements
* Faster training
* Smaller artifacts
* Multiple adapters can share a base model

---

## Q2. When would you fine-tune an LLM?

Good reasons:

* Consistent output format
* Specialized task behavior
* Domain-specific language patterns
* Instruction following
* Classification/extraction behavior

Bad reason:

> "The model doesn't know today's database contents."

That is usually a retrieval/data-access problem, not a fine-tuning problem.

---

# 11. LLM Evaluation

Evaluation should not rely only on "does the answer look good?"

## RAG Evaluation

Separate:

### Retrieval

* Recall@K
* Precision@K
* MRR
* NDCG

### Generation

* Faithfulness
* Answer relevance
* Context relevance
* Citation correctness
* Completeness

### Product metrics

* User satisfaction
* Resolution rate
* Latency
* Cost
* Escalation rate

---

## Critical Interview Question

> Your RAG chatbot accuracy dropped from 85% to 70%. How do you debug it?

Strong answer:

1. Identify affected query categories.
2. Compare retrieval metrics.
3. Inspect retrieved chunks.
4. Check embedding/model changes.
5. Check chunking changes.
6. Check query rewriting.
7. Check reranking.
8. Check context construction.
9. Evaluate LLM generation independently.
10. Compare against a fixed evaluation dataset.

---

# 12. Multimodality

## Questions

### Q1. What is multimodal AI?

Models/systems that work across modalities such as:

* Text
* Images
* Audio
* Video
* Documents

---

### Q2. How would you build a multimodal document assistant?

Possible architecture:

**Upload**

→ Document classification

→ OCR / parsing

→ Image extraction

→ Table extraction

→ Multimodal embeddings or representations

→ Vector/index storage

→ Retrieval

→ Multimodal LLM

→ Grounded response

---

# 13. Voice Agents

## Typical Architecture

**User Speech**

↓

**VAD**

↓

**Speech-to-Text**

↓

**LLM / Agent**

↓

**Tool Calls**

↓

**Text-to-Speech**

↓

**Audio Output**

---

## Key Metrics

Voice agents are extremely sensitive to latency.

Measure:

* Time to first transcript
* Time to first token
* Time to first audio
* End-to-end response latency
* Interruption handling
* STT accuracy
* TTS quality
* Tool-call latency
* Conversation completion rate

---

## Interview Question

> Why does a voice agent feel slow even if the LLM itself responds quickly?

Potential bottlenecks:

* Audio upload
* VAD
* STT
* Network latency
* LLM time-to-first-token
* Tool calls
* TTS
* Audio streaming

A strong candidate should think about **the entire latency budget**, not only LLM inference.

---

# 14. RAG Chatbot System Design

## Interview Prompt

> Design a production-grade enterprise RAG chatbot that supports millions of documents and thousands of concurrent users.

### Expected High-Level Architecture

**Client**

↓

**API Gateway / Load Balancer**

↓

**FastAPI Service**

↓

**Authentication + Rate Limiting**

↓

**Query Orchestrator**

↓

**Query Rewriting**

↓

**Parallel Retrieval**

→ Vector Search

→ Keyword Search

→ Metadata Filters

↓

**Reranker**

↓

**Context Builder**

↓

**LLM Gateway**

↓

**Streaming Response**

↓

**Citation / Response Validation**

↓

**Client**

---

# 15. Document Ingestion Architecture

Use an asynchronous pipeline:

**Upload**

→ Object Storage

→ Queue

→ Document Workers

→ Parser/OCR

→ Chunker

→ Embedding Workers

→ Vector DB

→ Metadata DB

→ Search Index

This separates ingestion from online query serving.

---

# 16. Scaling Questions

### Q1. What if there are 10 million documents?

Discuss:

* Distributed vector database
* Sharding
* Metadata filtering
* Index optimization
* Incremental ingestion
* Batch embedding
* Object storage
* Separate hot/cold data
* Caching

---

### Q2. What if 10,000 users query simultaneously?

Discuss:

* Horizontal API scaling
* Load balancing
* Async I/O
* Connection pooling
* Rate limiting
* Queueing
* LLM concurrency limits
* Streaming
* Caching
* Autoscaling
* Backpressure

---

### Q3. How would you reduce LLM cost?

Possible strategies:

* Semantic caching
* Smaller models for simple queries
* Query classification
* Reduce retrieved context
* Better chunking
* Prompt compression
* Batch inference
* Model routing
* Response caching
* Token budgeting

---

# 17. Production Reliability

Ask:

> What happens if the vector DB is down?

A strong design should have:

* Timeouts
* Retries with backoff
* Circuit breakers
* Graceful degradation
* Health checks
* Observability
* Alerts

Avoid unlimited retries.

---

# 18. Security Questions

For enterprise RAG, ask:

### How do you prevent users from retrieving documents they shouldn't see?

Expected concepts:

* Authentication
* Authorization
* Document-level ACLs
* Metadata filters
* Tenant isolation
* Access control at retrieval time
* Audit logs

Important:

**Do not rely on the LLM to enforce authorization.**

Authorization must happen before/within retrieval.

---

# 19. Observability

Track:

### API

* Requests/sec
* Error rate
* P50/P95/P99 latency

### Retrieval

* Recall
* Retrieved document distribution
* Reranker scores
* Empty retrieval rate

### LLM

* Input tokens
* Output tokens
* TTFT
* Generation latency
* Error rate
* Cost

### Product

* User feedback
* Resolution rate
* Abandonment
* Escalation

Use trace IDs to connect:

**User → API → Retrieval → Reranker → LLM → Tools → Response**

---

# 20. Senior-Level Scenario Questions

These are particularly useful for distinguishing strong candidates.

### Scenario 1

> The chatbot gives correct answers but takes 8 seconds. How do you bring latency below 2 seconds?

Look for:

* Distributed tracing
* Parallel retrieval
* Caching
* Faster embeddings
* Smaller/routed models
* Streaming
* KV caching
* Reduced context
* Faster reranking
* Connection pooling

---

### Scenario 2

> Retrieval is excellent, but the LLM still hallucinates.

Look for:

* Better grounding instructions
* Context quality
* Context compression
* Citation requirements
* Answerability detection
* Abstention
* Faithfulness evaluation
* Structured generation

---

### Scenario 3

> Retrieval is poor even though embeddings look good.

Investigate:

* Chunking
* Query formulation
* Metadata filters
* Hybrid retrieval
* Top-K
* Embedding normalization
* Domain vocabulary
* Reranking
* Evaluation dataset quality

---

### Scenario 4

> Your API works locally but becomes extremely slow under load.

Investigate:

* Blocking code
* Event-loop blocking
* DB connection pool
* Thread pool saturation
* CPU bottlenecks
* External API limits
* Network latency
* Container resource limits

---

# 21. Practical Coding Round

Give the candidate:

> Implement an async document-processing service that accepts documents, processes them concurrently, limits concurrency to 10, retries failed operations up to 3 times, and returns per-document status.

Evaluate:

* Python fundamentals
* `asyncio`
* Exception handling
* Concurrency control
* Retry logic
* Clean code
* Timeouts
* Resource management

Bonus:

> Add cancellation and graceful shutdown.

---

# 22. Practical RAG Round

Give:

> You have 50,000 company documents. Build a RAG chatbot that supports PDF, DOCX, and scanned PDFs.

Ask the candidate to design:

1. Ingestion.
2. OCR.
3. Chunking.
4. Embedding.
5. Vector storage.
6. Metadata.
7. Retrieval.
8. Reranking.
9. Prompt construction.
10. LLM inference.
11. Citations.
12. Evaluation.
13. Monitoring.
14. Security.
15. Scaling.

---

# 23. Rapid-Fire Questions

Use these toward the end.

1. What is the difference between concurrency and parallelism?
2. What is an event loop?
3. Why can `time.sleep()` be dangerous inside async code?
4. Threads vs processes?
5. What is connection pooling?
6. What is idempotency?
7. What is a circuit breaker?
8. What is backpressure?
9. What is batching?
10. What is tokenization?
11. What is perplexity?
12. What is temperature?
13. What is KV cache?
14. What is LoRA?
15. What is RAG?
16. What is hybrid search?
17. What is reranking?
18. What is hallucination?
19. What is grounding?
20. What is semantic caching?
21. What is prompt injection?
22. How do you secure RAG?
23. How do you evaluate retrieval?
24. How do you evaluate generation?
25. How do you reduce LLM latency?
26. How do you reduce LLM cost?
27. How do you scale vector search?
28. How do you handle model failures?
29. How do you stream an LLM response?
30. How do you monitor an AI application?

---

# 24. Evaluation Rubric

Score each area from **1–5**.

| Score | Meaning                                                      |
| ----- | ------------------------------------------------------------ |
| 1     | Cannot explain fundamentals                                  |
| 2     | Basic theoretical understanding                              |
| 3     | Solid working knowledge                                      |
| 4     | Strong practical + theoretical knowledge                     |
| 5     | Deep expertise; understands trade-offs and production issues |

### Suggested weighting

| Area                                 | Weight |
| ------------------------------------ | -----: |
| Python / Coding                      |    15% |
| ML Fundamentals                      |    10% |
| Deep Learning / Transformers         |    15% |
| FastAPI / Async / Backend            |    15% |
| LLM / RAG / Fine-tuning              |    20% |
| Multimodality / Voice                |    10% |
| System Design / Scaling              |    15% |

---

# 25. What a Strong Candidate Looks Like

A strong candidate should not merely name technologies.

They should be able to explain:

**Why → Trade-off → Failure mode → Alternative → Production implication**

For example, instead of:

> "We use vector search for RAG."

A strong answer sounds like:

> "I'd use hybrid retrieval because semantic search handles paraphrases while lexical retrieval is better for exact identifiers. I'd retrieve a relatively large candidate set, rerank it, apply ACL filtering, and then pass a small high-quality context to the LLM. I'd evaluate retrieval separately from generation so that a drop in answer quality can be localized."

That distinction is particularly important for **senior AI/backend interviews**.

---

# 26. Final Interview Flow

### Round 1 — Coding

Python + algorithms + backend problem.

### Round 2 — ML/DL

ML fundamentals + neural networks + Transformers.

### Round 3 — Backend

FastAPI + async + concurrency + APIs + databases.

### Round 4 — GenAI

LLMs + RAG + OCR + fine-tuning + evaluation.

### Round 5 — Advanced AI

Multimodality + voice agents + agents/tool use.

### Round 6 — System Design

Design a production RAG chatbot.

### Round 7 — Deep Dive

Choose one project from the candidate's resume and challenge every architectural decision.

## Final Decision

A strong **Senior AI/ML Backend Engineer** should demonstrate all four dimensions:

**Coding ability + ML understanding + production backend engineering + AI system design.**

The most important signal is whether they can reason about **trade-offs and failure modes**, rather than whether they can recite definitions.
