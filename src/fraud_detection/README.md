# 🛡️ Fraud Detection System

## Beginner-Friendly Technical Explanation

### 1. What is this project?

This project is a **machine-learning system that predicts whether a credit-card transaction is likely to be fraudulent**.

Think of it like a security guard checking every transaction.

When a transaction arrives, the system looks at information such as:

* 💰 How much money was spent
* 🕐 What time the transaction happened
* 🔄 How many transactions happened in the last hour
* 🌍 How risky the transaction's location appears
* 📊 Five additional numerical features (`v1` to `v5`)

The machine-learning model then gives the transaction a **fraud probability**.

For example:

```text
Fraud probability = 0.92
```

This means the model considers the transaction highly suspicious.

The system then converts that probability into a risk category:

```text
Probability < 25%       → LOW_RISK
25% to < 50%            → MEDIUM_RISK
50% or higher           → HIGH_RISK
```

So the basic flow is:

```text
Transaction
     ↓
Collect transaction information
     ↓
Random Forest model
     ↓
Fraud probability
     ↓
Risk score + risk category
     ↓
Return result
```

---

# 2. What does the system look at?

The model uses **9 input features**.

| Feature         | Simple meaning                                       |
| --------------- | ---------------------------------------------------- |
| `amount`        | Amount of money in the transaction                   |
| `time_hour`     | Hour when the transaction happened                   |
| `velocity_1h`   | Number of transactions made during the previous hour |
| `location_risk` | Risk score for the transaction's location            |
| `v1`            | Additional numerical pattern                         |
| `v2`            | Additional numerical pattern                         |
| `v3`            | Additional numerical pattern                         |
| `v4`            | Additional numerical pattern                         |
| `v5`            | Additional numerical pattern                         |

For example, a transaction might look like:

```text
amount         = $250
time_hour      = 2
velocity_1h    = 4
location_risk  = 0.85
v1             = 1.8
v2             = -0.5
v3             = 0.2
v4             = 1.1
v5             = -0.8
```

The model receives all of these values together.

---

# 3. Where does the data come from?

This is an important part of this particular project.

The code does **not load a real credit-card dataset**.

Instead, `generate_data()` creates **synthetic data**.

Synthetic data means:

> Data created artificially by the program for training and demonstration.

The default number of transactions is:

```python
n_samples = 1500
```

So the program creates **1,500 artificial transactions**.

---

# 4. How is the fake transaction data created?

The code creates different types of values using random mathematical distributions.

For example:

```python
amount = np.random.exponential(scale=100, size=n_samples)
```

This creates transaction amounts.

The idea is to produce many smaller transactions and fewer very large transactions.

For example, you might get:

```text
$12
$45
$72
$101
$350
$900
$2,000
...
```

---

### Transaction time

```python
time_hour = np.random.randint(0, 24, size=n_samples)
```

This generates a random hour between:

```text
0 → midnight
...
12 → noon
...
23 → 11 PM
```

---

### Transaction velocity

```python
velocity_1h = np.random.poisson(lam=2, size=n_samples)
```

This represents approximately how many transactions happened during the previous hour.

For example:

```text
0 transactions
1 transaction
2 transactions
3 transactions
5 transactions
...
```

A higher number can indicate unusual activity.

---

### Location risk

```python
location_risk = np.random.uniform(0, 1, size=n_samples)
```

This generates a number between 0 and 1.

Think of it as:

```text
0.05 → low location risk
0.30 → moderate risk
0.85 → high risk
0.95 → very high risk
```

This is also synthetic data; the code is not actually calculating geographical risk from GPS or an external location service.

---

# 5. What are `v1` to `v5`?

The code creates them like this:

```python
v1 = np.random.normal(0, 1, size=n_samples)
v2 = np.random.normal(0, 1, size=n_samples)
...
```

These are simply **additional numerical features generated randomly**.

They could represent hidden or anonymized patterns in a real-world dataset.

However, there is an important point:

> In this code, `v1` to `v5` are **not actually calculated using PCA**.

They are generated using a normal distribution.

So it would be inaccurate to say:

> "The system performs PCA."

A better description is:

> **"`v1` to `v5` are synthetic numerical features included to represent additional transaction patterns."**

---

# 6. How does the program decide which transactions are fraud?

This is one of the most important parts of the project.

The program creates something called a:

```python
fraud_score
```

The formula is:

```python
fraud_score = (
    0.015 * amount +
    1.2 * velocity_1h +
    2.5 * location_risk +
    0.8 * (v1 > 1.5) +
    1.0 * (time_hour < 4) +
    random_noise
)
```

Don't worry about the mathematical notation.

Think of it as a **risk-points system**.

The program says:

```text
Large transaction
        ↓
adds some risk

Many recent transactions
        ↓
adds more risk

Risky location
        ↓
adds more risk

v1 is unusually high
        ↓
adds more risk

Transaction happens before 4 AM
        ↓
adds more risk
```

Then the program adds a small amount of random noise so that the data isn't perfectly predictable.

---

# 7. How does it finally decide "fraud" or "not fraud"?

After calculating the fraud score for all 1,500 transactions, the code does this:

```python
is_fraud = (fraud_score > np.percentile(fraud_score, 88)).astype(int)
```

In simple language:

> **The transactions with fraud scores in approximately the highest 12% are labelled as fraud.**

So your synthetic dataset is intentionally created with roughly:

```text
88% → Not Fraud
12% → Fraud
```

This is important because it means your dataset has **class imbalance**, but it is not the extreme <1% fraud rate often seen in real-world credit-card systems.

For this project, it's approximately **12% fraud**.

---

# 8. What happens during training?

Once the synthetic data has been generated, the program separates the data into:

### X — the information about the transaction

```python
X = df.drop(columns=["is_fraud"])
```

This contains:

```text
amount
time_hour
velocity_1h
location_risk
v1
v2
v3
v4
v5
```

### y — the answer

```python
y = df["is_fraud"]
```

This contains:

```text
0 → Not Fraud
1 → Fraud
```

So you can think of it as:

```text
X = Questions / Evidence

y = Correct Answer
```

---

# 9. Why do we split the data?

The code uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y
)
```

This divides the dataset into two parts.

```text
1,500 transactions
       ↓
 ┌───────────────┐
 │               │
 ↓               ↓
80%             20%
Training        Testing
~1,200          ~300
```

### Training data

The model learns from the training data.

### Testing data

The model does **not** train on the test data.

Instead, we use it to ask:

> "After learning, can the model correctly identify transactions it hasn't seen before?"

---

# 10. What does `stratify=y` do?

This is useful because we have two classes:

```text
Fraud
Not Fraud
```

We want the training and testing datasets to have roughly the same fraud/non-fraud proportions.

For example:

```text
Original dataset
≈ 88% normal
≈ 12% fraud

        ↓

Training
≈ 88% normal
≈ 12% fraud

Testing
≈ 88% normal
≈ 12% fraud
```

That's what `stratify=y` helps achieve.

---

# 11. What is Random Forest?

Now we reach the actual machine-learning model.

Your code uses:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=RANDOM_SEED
)
```

The easiest way to understand Random Forest is:

> **It is a collection of decision trees that work together.**

Imagine one decision tree asking:

```text
Is the amount high?
       ↓
      YES
       ↓
Is location risk high?
       ↓
      YES
       ↓
Were there many recent transactions?
       ↓
      YES
       ↓
Suspicious!
```

Another tree might make a different set of decisions.

Another tree might look more closely at the transaction time.

Another might use `v1`.

You have **100 trees**.

They all contribute to the final prediction.

That's why it is called a:

> **Random Forest**

A forest contains many trees.

---

# 12. Why use 100 trees?

Your code says:

```python
n_estimators=100
```

This means:

> **Build 100 decision trees.**

Using multiple trees generally makes the prediction more stable than relying on a single decision tree.

For this project, 100 is a reasonable model setting.

---

# 13. What does `max_depth=8` mean?

Each tree can only grow to a maximum depth of 8 levels.

Imagine:

```text
Question 1
    ↓
Question 2
    ↓
Question 3
    ↓
...
    ↓
Question 8
```

Limiting the depth helps prevent individual trees from becoming excessively complicated and memorizing the training data.

In simple terms:

> **`max_depth=8` puts a limit on how complicated each decision tree can become.**

---

# 14. How does the model learn?

This line is where the learning happens:

```python
model.fit(X_train, y_train)
```

The model sees examples such as:

```text
Amount   Velocity   Location Risk   Time   Fraud
-------------------------------------------------
$20      1          0.1             14     NO
$80      2          0.2             10     NO
$900     5          0.8              2     YES
$4000    7          0.9              3     YES
```

It tries to discover patterns that separate:

```text
NOT FRAUD
```

from:

```text
FRAUD
```

After training, the Random Forest has learned patterns from the training data.

---

# 15. How do we know whether the model is good?

After training, the model is tested using the test dataset.

The code makes two kinds of predictions:

```python
y_pred = model.predict(X_test)
```

and:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

These are slightly different.

### `predict()`

Gives the final class:

```text
0 → Not Fraud
1 → Fraud
```

### `predict_proba()`

Gives the probability of fraud.

For example:

```text
0.08
0.23
0.61
0.94
```

This probability is very useful because your system later uses it to determine risk.

---

# 16. What is ROC-AUC?

Your code calculates:

```python
roc_auc_score(y_test, y_prob)
```

ROC-AUC measures how well the model can distinguish between:

```text
Fraud
```

and:

```text
Not Fraud
```

A simple way to think about it is:

> **Can the model generally give higher fraud probabilities to fraudulent transactions than to legitimate ones?**

A score closer to 1 is generally better.

For example:

```text
0.50 → little/no useful separation
0.70 → reasonable
0.80 → good
0.90+ → very strong separation
1.00 → perfect separation
```

But remember:

> **A high ROC-AUC on this synthetic dataset does not mean the model will automatically perform equally well on real banking data.**

---

# 17. What is F1 score?

The code also calculates:

```python
f1_score(y_test, y_pred)
```

F1 score is useful when we care about correctly identifying the fraud class.

It combines two ideas:

### Precision

When the model says:

> "This is fraud."

How often is it actually fraud?

### Recall

Of all the transactions that really are fraud:

> "How many did the model successfully find?"

F1 combines these two into one score.

So:

> **F1 helps us understand how well the model balances finding fraud while avoiding too many incorrect fraud predictions.**

---

# 18. What happens after training?

The trained model is stored in:

```python
self.model = {
    "classifier": model,
    "feature_names": list(X.columns),
    "metrics": {
        "roc_auc": ...,
        "f1_score": ...
    }
}
```

Then:

```python
self.save()
```

saves the model using the artifact name:

```text
fraud_detection_model.joblib
```

This means we don't have to train the model every time somebody wants to make a prediction.

We can train it once, save it, and later load it for predictions.

---

# 19. What happens when a new transaction arrives?

This is the **prediction stage**.

Suppose the API receives:

```text
Amount         = $250
Time           = 2 AM
Velocity       = 4
Location risk  = 0.85
V1             = 1.8
V2             = -0.5
V3             = 0.2
V4             = 1.1
V5             = -0.8
```

The `predict()` function receives those values.

If the model hasn't already been loaded:

```python
if self.model is None:
    self.load()
```

it loads the saved model.

---

# 20. Why does the code use `feature_names`?

The model was trained using this exact order:

```text
amount
time_hour
velocity_1h
location_risk
v1
v2
v3
v4
v5
```

The prediction code uses:

```python
df_input = pd.DataFrame([input_data])[feature_names]
```

This makes sure the incoming data follows the same feature structure the model was trained with.

In simple terms:

> **The model expects the same inputs it saw during training.**

---

# 21. How does the model make the final prediction?

This line is very important:

```python
prob = float(model_cls.predict_proba(df_input)[0, 1])
```

Suppose the model returns:

```text
0.92
```

That means:

```text
Fraud probability = 92%
```

Then your code converts that into a risk score:

```python
risk_score = int(round(prob * 100))
```

So:

```text
0.92 × 100 = 92
```

Therefore:

```text
risk_score = 92
```

---

# 22. How is the risk category decided?

Your code uses:

```python
status = (
    "HIGH_RISK" if prob >= 0.5
    else "MEDIUM_RISK" if prob >= 0.25
    else "LOW_RISK"
)
```

So:

```text
             Fraud Probability

                  92%
                   │
                   ▼
             HIGH_RISK 🔴
```

The rules are:

```text
0% ───────── 25% ───────── 50% ───────── 100%
       LOW          MEDIUM         HIGH
```

More precisely:

```text
< 0.25       → LOW_RISK
0.25–<0.50   → MEDIUM_RISK
≥ 0.50       → HIGH_RISK
```

These thresholds are **rules chosen by your application**. They are not automatically learned by the Random Forest.

---

# 23. What are `top_risk_factors`?

The model also calculates:

```python
model_cls.feature_importances_
```

This tells us which features were generally important to the Random Forest.

The code selects the top three:

```python
top_indices = np.argsort(importances)[::-1][:3]
```

and returns their names.

For example:

```json
"top_risk_factors": [
    "location_risk",
    "velocity_1h",
    "amount"
]
```

This means:

> **These features were among the most important features for the trained Random Forest overall.**

Important clarification:

This does **not** necessarily mean:

> "These three features caused this specific transaction to be classified as fraud."

They are global model feature importances.

For a more precise explanation of an individual prediction, you would need a local explanation technique such as SHAP.

---

# 24. What does the final output look like?

Your `predict()` function returns:

```json
{
    "fraud_probability": 0.92,
    "risk_score": 92,
    "alert_status": "HIGH_RISK",
    "top_risk_factors": [
        "location_risk",
        "velocity_1h",
        "amount"
    ]
}
```

A human can read this as:

> **The model estimates a 92% fraud probability, so the transaction receives a risk score of 92 and is classified as HIGH_RISK. The most important features in the model are location risk, transaction velocity, and transaction amount.**

---

# 25. The entire project in one example

Imagine a customer makes this transaction:

```text
💳 Amount: $2,000
🕐 Time: 2 AM
🔄 Transactions in last hour: 6
🌍 Location risk: 0.9
```

The system gives these values to the Random Forest.

The Random Forest contains 100 decision trees.

The trees analyze the transaction and collectively produce:

```text
Fraud probability = 0.87
```

Your code then calculates:

```text
0.87 × 100 = 87
```

So:

```text
Risk score = 87
```

Since:

```text
87% ≥ 50%
```

the result becomes:

```text
HIGH_RISK
```

The system returns the result.

So the whole process is:

```text
              💳 Transaction
                    ↓
       ┌────────────────────────┐
       │ Transaction information│
       │                        │
       │ Amount                 │
       │ Time                   │
       │ Velocity               │
       │ Location risk          │
       │ V1 - V5                │
       └───────────┬────────────┘
                   ↓
          🌲 Random Forest
          100 decision trees
                   ↓
          Fraud probability
              e.g. 87%
                   ↓
          ┌────────┼────────┐
          ↓        ↓        ↓
        <25%    25-50%     ≥50%
          ↓        ↓        ↓
        LOW     MEDIUM     HIGH
```

---

# 26. What is actually implemented vs. what is not?

This is especially important if you're going to explain this project in an interview.

### ✅ Your code actually implements

* Synthetic transaction data generation
* 1,500 default training samples
* Fraud-label generation
* Train/test split
* Stratified splitting
* Random Forest classification
* 100 decision trees
* Maximum tree depth of 8
* Fraud probability prediction
* Risk score calculation
* Three risk categories
* ROC-AUC evaluation
* F1 evaluation
* Model serialization/loading
* Global feature importance
* Deterministic randomness through `RANDOM_SEED`

### ❌ Your shown code does not implement

The original description mentioned several things that are **not shown in this code**:

* Actual PCA transformation
* Formal probability calibration
* Real credit-card transaction data
* Real geographical risk calculation
* <50 ms latency measurement
* KS-test drift monitoring
* Concept-drift monitoring
* Automatic retraining
* A fallback rule engine
* Blocking transactions
* 2FA integration
* API endpoint implementation

Those could be **future production enhancements**, but you shouldn't say they're currently implemented unless they're present elsewhere in your project.

---

# 27. The project architecture

Based specifically on the code you provided, the architecture is better described as:

```text
                 TRAINING
                    │
                    ▼
          generate_data()
                    │
                    ▼
        1,500 synthetic transactions
                    │
                    ▼
          Create fraud labels
                    │
                    ▼
          Train/Test Split
              /          \
             /            \
            ▼              ▼
       Training          Testing
          data             data
            │               │
            ▼               │
     Random Forest           │
       100 trees             │
            │                │
            └───────┬────────┘
                    ▼
              Evaluate model
                    │
             ROC-AUC + F1
                    │
                    ▼
             Save model
                    │
                    ▼
       fraud_detection_model.joblib


                 PREDICTION
                    │
                    ▼
            New transaction
                    │
                    ▼
              Load model
                    │
                    ▼
           Random Forest
                    │
                    ▼
          Fraud probability
                    │
                    ▼
             Risk score
                    │
                    ▼
       LOW / MEDIUM / HIGH
                    │
                    ▼
                Response
```

---

# 28. How I would explain the project in an interview

If someone asks:

### "Tell me about your fraud detection project."

You can say:

> **"I built a machine-learning fraud detection pipeline using a Random Forest classifier. Since this is a prototype, I generated 1,500 synthetic transactions using features such as transaction amount, transaction time, transaction velocity, location risk, and five additional numerical features.**
>
> **I generated fraud labels using a risk-scoring formula where factors such as high transaction amounts, high transaction velocity, risky locations, unusual feature values, and transactions occurring before 4 AM increase the fraud score. The highest-risk approximately 12% of transactions were labelled as fraud.**
>
> **I then split the data into 80% training and 20% testing using stratification to preserve the fraud ratio. I trained a Random Forest with 100 trees and a maximum depth of 8.**
>
> **For evaluation, I used ROC-AUC to measure how well the model separates fraudulent and legitimate transactions, and F1-score to evaluate the balance between precision and recall for fraud detection.**
>
> **For inference, the model produces a fraud probability. I convert that probability into a 0–100 risk score and classify the transaction as LOW_RISK, MEDIUM_RISK, or HIGH_RISK using thresholds of 25% and 50%. I also return the three globally most important features from the Random Forest."**

---

# 29. The simplest possible explanation

If you forget everything else, remember this:

> **I created fake transaction data, taught a Random Forest what suspicious transactions look like, and then used that trained model to estimate the probability that a new transaction is fraudulent.**
>
> **If the probability is below 25%, I call it low risk. Between 25% and 50%, it's medium risk. Above 50%, it's high risk.**
>
> **I evaluate the model using ROC-AUC and F1-score and save the trained model so it can be used later for predictions.**

That's the project.

You don't need to memorize the code line-by-line. **Understand this flow first, and then every line of the code has a reason for being there.**
