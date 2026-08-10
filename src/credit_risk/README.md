# 💳 Credit Risk Prediction System

## Beginner-Friendly Explanation

## 1. What is this project?

This project is a **machine-learning system that estimates whether a person is likely to default on a loan**.

In simple words:

> **A bank wants to decide: “If we give this person a loan, how likely are they to not repay it?”**

Your system looks at information about the borrower, such as:

* Credit score
* Annual income
* Debt-to-income ratio
* Loan amount
* Previous delinquencies
* Employment history

Then the machine-learning model calculates something like:

```text
Default probability = 12%
```

That means:

> The model estimates that this borrower has a 12% probability of defaulting.

Your program then turns that probability into a **risk tier** and an **underwriting decision**.

For example:

```text
12% default probability
        ↓
AA (Near Prime)
        ↓
APPROVED
```

So the basic idea is:

```text
Borrower information
        ↓
Prepare the data
        ↓
Logistic Regression
        ↓
Default probability
        ↓
Risk tier
        ↓
Loan decision
```

---

# 2. What does "default" mean?

Before going further, understand this word.

**Default** means the borrower fails to meet their loan repayment obligations.

For example:

Imagine someone borrows:

```text
$25,000
```

and then fails to make the required payments.

That borrower may eventually be considered a **default**.

So your model is trying to predict:

```text
Will this borrower default?
```

The answer is represented as:

```text
0 → No default
1 → Default
```

---

# 3. What information does the model use?

Your model uses **6 features**.

| Feature             | Simple meaning                                                    |
| ------------------- | ----------------------------------------------------------------- |
| `credit_score`      | How strong the person's credit history is                         |
| `annual_income`     | How much money they earn per year                                 |
| `dti_ratio`         | How much of their income is already committed to debt             |
| `loan_amount`       | How much money they want to borrow                                |
| `delinquencies_2yr` | How many times they were late on payments during the last 2 years |
| `employment_years`  | How long they have been employed                                  |

Think of these as the questions the model asks about the borrower.

---

# 4. Let's understand each feature

### Credit score

Your code generates scores between:

```text
300 → 850
```

Generally:

```text
Higher credit score
        ↓
Lower expected default risk
```

For example:

```text
750 → generally stronger credit profile
500 → generally weaker credit profile
```

Your synthetic model specifically makes higher credit scores reduce default risk.

---

### Annual income

Your code generates income between:

```text
$20,000 → $180,000
```

Generally, higher income can mean the borrower has more ability to repay.

Your synthetic formula therefore makes higher income reduce default risk.

---

### DTI ratio

DTI means:

> **Debt-to-Income ratio**

This tells us how much of someone's income is already being used for debt obligations.

For example:

```text
Annual income = $60,000
Debt obligations = $24,000

DTI = 24,000 / 60,000
    = 0.40
    = 40%
```

So:

```text
DTI = 40%
```

means roughly 40% of the borrower's income is going toward debt obligations.

In your synthetic model:

> **Higher DTI → higher default risk.**

---

### Loan amount

This is simply:

> **How much money is the person asking to borrow?**

Your synthetic data generates loan amounts between:

```text
$2,000 → $50,000
```

Your model treats larger loan amounts as increasing default risk.

---

### Delinquencies

A delinquency means the borrower was late with a payment.

For example:

```text
0 delinquencies → no recent late payments
1 delinquency   → one late payment
3 delinquencies → three late payments
```

Your model assumes:

> **More previous delinquencies → higher default risk.**

---

### Employment years

This represents how long the person has been employed.

Your synthetic model assumes:

> **More employment experience → lower default risk.**

Again, this is the relationship you've intentionally created in your synthetic data.

---

# 5. Where does the data come from?

Just like your Fraud Detection project, this project uses **synthetic data**.

The code creates:

```python
generate_data(n_samples=1500)
```

So the program generates:

> **1,500 artificial borrowers.**

It does not currently load real bank/customer loan data.

For example, the generated dataset might conceptually look like:

```text
Credit Score | Income | DTI | Loan | Delinquencies | Employment | Default
---------------------------------------------------------------------------
750          | $80k   | .20 | $10k | 0             | 8           | 0
580          | $45k   | .50 | $30k | 2             | 2           | 1
720          | $70k   | .25 | $15k | 0             | 6           | 0
500          | $35k   | .55 | $40k | 3             | 1           | 1
```

Here:

```text
0 = did not default
1 = defaulted
```

---

# 6. How does the program create the "default" label?

This is one of the most important parts to understand.

The code calculates a value called:

```python
logit
```

Don't worry about the name.

Think of it as:

> **A mathematical risk score.**

The formula considers:

```text
Credit score
Income
DTI
Loan amount
Previous delinquencies
Employment years
```

For example:

```text
Higher credit score
        ↓
reduces risk

Higher income
        ↓
reduces risk

Higher DTI
        ↓
increases risk

Larger loan
        ↓
increases risk

More delinquencies
        ↓
increases risk

More employment years
        ↓
reduces risk
```

The code combines all these effects into one number.

---

# 7. What does this scary formula mean?

Your code has:

```python
logit = (
    -0.012 * (credit_score - 600) +
    -0.00002 * (annual_income - 50000) +
    4.5 * dti_ratio +
    0.00005 * loan_amount +
    0.8 * delinquencies_2yr +
    -0.08 * employment_years +
    noise
)
```

You don't need to memorize the equation.

Understand the signs:

```text
Feature                    Effect in your synthetic data

Credit score ↑             Risk ↓
Income ↑                   Risk ↓
DTI ↑                      Risk ↑
Loan amount ↑              Risk ↑
Delinquencies ↑            Risk ↑
Employment years ↑         Risk ↓
```

The numbers in front of each feature determine **how strongly that feature affects the synthetic risk score**.

For example:

```text
4.5 * dti_ratio
```

means DTI has a relatively strong positive contribution to the risk score.

---

# 8. What is the next step?

After calculating the `logit`, your code converts it into a probability:

```python
prob_default = 1 / (1 + np.exp(-logit))
```

This is called the **sigmoid function**.

Its job is simple:

> **Turn the model's mathematical score into a number between 0 and 1.**

For example:

```text
Risk calculation
      ↓
Sigmoid function
      ↓
0.12
```

Then:

```text
0.12 × 100 = 12%
```

So we can say:

> **Estimated default probability = 12%.**

---

# 9. How is the actual default label created?

Your code then does:

```python
is_default = (
    np.random.uniform(0, 1, size=n_samples) < prob_default
).astype(int)
```

In simple terms:

The program generates a random number between 0 and 1.

If that random number is below the calculated default probability:

```text
→ Default = 1
```

Otherwise:

```text
→ Default = 0
```

For example:

```text
Calculated probability = 0.70

Random number = 0.30

0.30 < 0.70

→ Default
```

But:

```text
Calculated probability = 0.20

Random number = 0.70

0.70 < 0.20

→ No default
```

This gives the synthetic dataset some randomness.

---

# 10. Why do we need training data?

Now we have 1,500 artificial borrowers.

The program separates them into:

```text
X = borrower information

y = whether they defaulted
```

So:

```text
X:
Credit score
Income
DTI
Loan amount
Delinquencies
Employment

y:
0 or 1
```

You can think of it as:

```text
X = Evidence

y = Answer
```

---

# 11. Why split the dataset?

The code uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y
)
```

This creates:

```text
1,500 borrowers
       ↓
 ┌───────────────┐
 │               │
 ↓               ↓
80%             20%
Training        Testing
~1,200          ~300
```

### Training data

The model learns from these borrowers.

### Testing data

The model hasn't trained on these borrowers.

We use them to see whether the model learned something useful.

It's like studying for an exam:

```text
Training data = practice questions

Test data = actual exam
```

---

# 12. Why use `StandardScaler`?

This is one of the most confusing parts of the code.

Look at the feature sizes:

```text
Credit score       → 300–850
Annual income      → 20,000–180,000
DTI ratio          → 0.05–0.60
Loan amount        → 2,000–50,000
```

These numbers are on completely different scales.

For example:

```text
Income = 80,000
DTI    = 0.30
```

Without scaling, the numbers have very different numerical sizes.

So the code uses:

```python
scaler = StandardScaler()
```

StandardScaler transforms the features so they are centered around zero and have a similar scale.

You can think of it as:

> **Putting all the features onto a comparable measuring scale before giving them to Logistic Regression.**

For example, conceptually:

```text
Before:

Income      = 80,000
Loan        = 25,000
Credit      = 640
DTI         = 0.42

             ↓ StandardScaler

After:

Income      = some standardized value
Loan        = some standardized value
Credit      = some standardized value
DTI         = some standardized value
```

The exact values aren't important for understanding the system.

---

# 13. Why must the scaler be saved?

This is very important.

During training, the scaler learns the training data's averages and standard deviations.

Then, when a new borrower arrives, you must transform their data using **the same scaler**.

That's why your saved model contains:

```python
"scaler": scaler
```

Later, during prediction:

```python
X_scaled = scaler.transform(df_input)
```

So the process is:

```text
Training:

Training data
     ↓
Learn scaling information
     ↓
Scale data
     ↓
Train model


Prediction:

New borrower
     ↓
Use SAME scaler
     ↓
Scale borrower
     ↓
Give to model
```

You shouldn't create a new scaler for every new borrower.

---

# 14. What is Logistic Regression?

This is the machine-learning algorithm you've chosen.

The name is slightly misleading.

Despite the word "regression", you're using it for **classification**.

Your question is:

```text
Will borrower default?
```

That's a classification problem.

The model learns relationships between the borrower features and the default outcome.

For example, it may learn patterns such as:

```text
Low credit score
+
High DTI
+
Previous delinquencies
        ↓
Higher default probability
```

Whereas:

```text
High credit score
+
Low DTI
+
Stable employment
        ↓
Lower default probability
```

---

# 15. Why is Logistic Regression useful here?

One major advantage is that the model is relatively easy to understand.

It learns a coefficient for each feature.

Conceptually:

```text
Feature                Effect

Credit score           ↓ risk
Income                 ↓ risk
DTI                    ↑ risk
Loan amount            ↑ risk
Delinquencies          ↑ risk
Employment             ↓ risk
```

So compared with a very complicated model, it's easier to inspect how the inputs relate to the prediction.

That's one reason Logistic Regression is commonly useful as a baseline for risk-scoring problems.

---

# 16. What happens during training?

This line performs the actual learning:

```python
model.fit(X_train_scaled, y_train)
```

The model sees many examples:

```text
Borrower A
Credit score = 750
DTI = 0.20
Delinquencies = 0
        ↓
Default = NO


Borrower B
Credit score = 520
DTI = 0.50
Delinquencies = 2
        ↓
Default = YES
```

The model tries to learn the mathematical relationship between the features and default.

After training, it can look at a new borrower and estimate:

```text
Probability of default
```

---

# 17. What does `predict_proba()` do?

This line:

```python
prob = float(model_cls.predict_proba(X_scaled)[0, 1])
```

asks the model:

> **"What is the probability that this borrower belongs to the default class?"**

For example:

```text
0.04 → 4%
0.12 → 12%
0.30 → 30%
0.70 → 70%
```

This is the key output of your model.

---

# 18. What happens to that probability?

Your code then puts the borrower into one of four risk tiers.

### 🟢 Less than 10%

```text
Probability < 0.10
```

Result:

```text
AAA (Prime)
APPROVED
```

Meaning:

> Very low predicted default risk.

---

### 🟢 10% to less than 25%

```text
0.10 ≤ Probability < 0.25
```

Result:

```text
AA (Near Prime)
APPROVED
```

Meaning:

> Some predicted risk, but still approved according to your rules.

---

### 🟡 25% to less than 45%

```text
0.25 ≤ Probability < 0.45
```

Result:

```text
B (Subprime)
MANUAL_REVIEW
```

Meaning:

> Risk is high enough that the system doesn't automatically approve or reject. A human should review the application.

---

### 🔴 45% or higher

```text
Probability ≥ 0.45
```

Result:

```text
CCC (High Risk)
REJECTED
```

Meaning:

> The predicted default risk is high enough for your current rules to reject the application.

---

# 19. Put the whole decision system together

Suppose someone applies for a loan.

They provide:

```text
Credit score       = 640
Annual income      = $55,000
DTI                = 0.42
Loan amount        = $25,000
Delinquencies      = 1
Employment         = 3 years
```

Your system does:

```text
Borrower information
        ↓
StandardScaler
        ↓
Logistic Regression
        ↓
Default probability
        ↓
Suppose: 12%
        ↓
12% is between 10% and 25%
        ↓
AA (Near Prime)
        ↓
APPROVED
```

So the final answer is:

```text
Default probability: 12%
Risk tier: AA (Near Prime)
Decision: APPROVED
```

---

# 20. What does ROC-AUC measure?

Your code calculates:

```python
roc_auc_score(y_test, y_prob)
```

ROC-AUC answers a general question:

> **How well can the model distinguish borrowers who default from borrowers who don't?**

Think of it as the model's ability to rank risky borrowers above safer borrowers.

Generally:

```text
0.50 → little useful separation
0.70 → reasonable
0.80 → good
0.90+ → very strong
1.00 → perfect
```

If your model gets an AUC around 0.88 on the synthetic test data, that's a strong result **on that dataset**.

But don't say:

> "My model is 88% accurate."

That's incorrect.

AUC and accuracy are different metrics.

---

# 21. What is accuracy?

Your code also calculates:

```python
accuracy_score(y_test, y_pred)
```

Accuracy asks:

> **Out of all test borrowers, how many did the model classify correctly?**

For example:

```text
100 borrowers tested

79 correctly classified
21 incorrectly classified

Accuracy = 79%
```

That's what an accuracy of approximately 0.79 would mean.

---

# 22. An important detail about your accuracy

There's a subtle point in your code.

You calculate:

```python
y_pred = model.predict(X_test_scaled)
```

Logistic Regression's `predict()` uses its standard classification threshold, effectively:

```text
Probability ≥ 50%
        ↓
Default

Probability < 50%
        ↓
No default
```

But your **underwriting system uses completely different thresholds**:

```text
10%
25%
45%
```

So the accuracy you're reporting is based on a **50% classification threshold**, not your actual underwriting thresholds.

That's okay for evaluating the classifier, but you should understand the distinction.

---

# 23. What happens when the model is used in production?

After training, your code stores:

```python
self.model = {
    "scaler": scaler,
    "classifier": model,
    "feature_names": list(X.columns),
    "metrics": ...
}
```

Then:

```python
self.save()
```

saves the model artifact:

```text
credit_risk_model.joblib
```

This contains both:

```text
Scaler
   +
Logistic Regression model
   +
Feature names
   +
Evaluation metrics
```

Why save the scaler too?

Because the new data must be transformed using the **same scaling process used during training**.

---

# 24. What happens when a new loan application arrives?

The `predict()` method handles it.

First:

```python
if self.model is None:
    self.load()
```

If the model isn't currently loaded, it loads the saved model.

Then the code gets the expected features:

```python
feature_names = self.model["feature_names"]
```

and prepares the incoming borrower data.

Then:

```python
X_scaled = scaler.transform(df_input)
```

scales it using the saved scaler.

Then:

```python
prob = model_cls.predict_proba(X_scaled)[0, 1]
```

gets the default probability.

Finally, your threshold rules determine:

```text
Risk tier
+
Underwriting decision
```

---

# 25. Your entire project in one diagram

```text
                 👤 BORROWER
                     │
                     ↓
          ┌─────────────────────┐
          │ Borrower Information│
          │                     │
          │ Credit score        │
          │ Income              │
          │ DTI                 │
          │ Loan amount         │
          │ Delinquencies       │
          │ Employment          │
          └──────────┬──────────┘
                     ↓
              StandardScaler
                     ↓
            Logistic Regression
                     ↓
             Default Probability
                     ↓
          ┌──────────┼───────────┐
          ↓          ↓           ↓
        <10%      10–25%      25–45%      ≥45%
          ↓          ↓           ↓          ↓
         AAA        AA           B         CCC
          ↓          ↓           ↓          ↓
      APPROVED   APPROVED   MANUAL REVIEW  REJECTED
```

That's your credit-risk system.

---

# 26. How this differs from your Fraud Detection project

This is a useful comparison.

### Fraud Detection

Question:

> **"Does this transaction look fraudulent?"**

Model:

```text
Random Forest
```

Output:

```text
Fraud probability
```

Decision:

```text
LOW / MEDIUM / HIGH
```

---

### Credit Risk

Question:

> **"How likely is this borrower to default on their loan?"**

Model:

```text
Logistic Regression
```

Output:

```text
Default probability
```

Decision:

```text
AAA / AA / B / CCC
```

Then:

```text
APPROVED
MANUAL_REVIEW
REJECTED
```

So you can remember:

```text
Fraud Detection
→ Is this transaction suspicious?

Credit Risk
→ Is this borrower likely to default?
```

---

# 27. What I would NOT say about this project

Your original documentation contains some statements that are too strong or aren't supported by the code.

### ❌ Don't say:

> "Regulations require credit models to be 100% explainable."

That's too absolute.

Real credit models can be subject to significant regulatory, legal, governance, fairness, documentation, validation, and adverse-action requirements, but "100% explainable" is not a precise universal regulatory requirement.

---

### ❌ Don't say:

> "Logistic Regression automatically produces calibrated probabilities."

Logistic Regression often produces useful probabilities, but **model calibration should be evaluated rather than assumed**.

Your code does not perform a separate calibration procedure.

---

### ❌ Don't say:

> "AAA, AA, B and CCC are regulatory risk tiers."

In your code, these are **your application's custom labels**.

They are not automatically regulatory ratings.

Your code explicitly defines:

```python
"AAA (Prime)"
"AA (Near Prime)"
"B (Subprime)"
"CCC (High Risk)"
```

Those are your decision categories.

---

### ❌ Don't say:

> "The system automatically generates regulatory adverse-action reasons."

Your current `predict()` function does not do that.

It returns:

```text
default probability
risk tier
underwriting decision
credit score
```

It doesn't generate explanations such as:

> "Rejected because DTI was high."

That could be a future enhancement.

---

### ❌ Don't say:

> "PSI monitoring is implemented."

Your shown code does not contain PSI monitoring.

You can say:

> **"PSI-based drift monitoring could be added as a production enhancement."**

---

# 28. What is actually implemented?

Based on the code you provided:

### ✅ Implemented

* Synthetic borrower data generation
* 1,500 default training samples
* Six borrower features
* Synthetic default probability generation
* Synthetic default labels
* Train/test split
* Stratification
* StandardScaler
* Logistic Regression
* Default probability prediction
* ROC-AUC evaluation
* Accuracy evaluation
* Model + scaler serialization
* Four risk tiers
* Automated underwriting decision
* Feature-name preservation

### ❌ Not implemented in the shown code

* Real borrower data
* Real credit bureau integration
* Formal probability calibration
* Actual regulatory scorecard/bucketing
* Adverse-action reason generation
* PSI monitoring
* Automatic recalibration
* Fairness testing
* Bias monitoring
* Human-review workflow
* Real loan approval/rejection integration

These can be described as **future production improvements**, but not as things the current code already does.

---

# 29. How I would explain this in an interview

If an interviewer asks:

### "Tell me about your credit-risk project."

Say:

> **"I built a credit-risk prediction pipeline using Logistic Regression. The goal is to estimate the probability that a borrower will default on a loan.**
>
> **For this prototype, I generated 1,500 synthetic borrower records containing credit score, annual income, debt-to-income ratio, loan amount, previous delinquencies, and employment history. I generated synthetic default labels based on relationships between those financial features and default risk.**
>
> **I split the data into 80% training and 20% testing using stratification. Because the features have very different numerical scales, I used StandardScaler before training the Logistic Regression model.**
>
> **The model outputs a probability of default rather than just a yes-or-no prediction. I evaluate the model using ROC-AUC and accuracy. During inference, I map the predicted probability into four application-specific risk tiers: AAA, AA, B, and CCC, which correspond to approved, approved, manual review, and rejected decisions respectively.**
>
> **I save both the trained Logistic Regression model and the scaler so that new applicants are processed using exactly the same transformations used during training."**

---

# 30. The simplest possible explanation

If you need to explain it to someone who knows **nothing about machine learning**, say:

> **"Imagine a bank receives a loan application. My system looks at the person's credit score, income, existing debt, requested loan amount, previous late payments, and employment history.**
>
> **A Logistic Regression model looks at those factors and estimates the probability that the person will fail to repay the loan.**
>
> **For example, if it predicts a 12% chance of default, my rules put the applicant into the AA risk tier and approve the application. If the predicted risk is much higher, the system can send the application for manual review or reject it.**
>
> **The model learns these patterns from 1,500 synthetic borrower examples, and I evaluate how well it distinguishes borrowers who default from those who don't."**

### The one thing to remember

Your Credit Risk project is basically:

**Borrower → Financial information → Logistic Regression → Default probability → Risk tier → Loan decision**

And your Fraud project is:

**Transaction → Transaction information → Random Forest → Fraud probability → Risk tier**

Once those two flows are clear, the rest of the code becomes much easier to understand.
