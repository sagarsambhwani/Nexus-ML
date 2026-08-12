# 📉 Customer Churn Prediction System

## Beginner-Friendly Explanation

## 1. What is this project?

This project tries to answer a simple business question:

> **"Which customers are likely to stop using our service?"**

When a customer cancels their subscription or stops being a customer, that is called **churn**.

For example:

```text
Customer A
Has been with us for 3 years
Few support problems
Long-term contract
        ↓
Probably won't leave
```

But:

```text
Customer B
Has been with us for 2 months
Many support tickets
Month-to-month contract
        ↓
May leave soon
```

Your system tries to identify customers like **Customer B** before they actually leave.

Then it recommends an action to try to retain them.

So the whole project is:

```text
Customer information
        ↓
Machine-learning model
        ↓
Churn probability
        ↓
Churn risk level
        ↓
Recommended retention action
```

---

# 2. What does "churn" mean?

**Churn means a customer leaves.**

Imagine a company has:

```text
10,000 customers
```

During the month:

```text
500 customers cancel
```

Those 500 customers are called **churned customers**.

A churn prediction system tries to identify customers who are likely to become part of that 500 **before they leave**.

Why?

Because if the company knows someone is at risk, it can try to keep them.

For example:

```text
Customer looks likely to leave
        ↓
Offer discount
        ↓
Solve support problem
        ↓
Give loyalty benefit
        ↓
Customer stays
```

---

# 3. What information does your model use?

Your model uses **6 features**:

| Feature             | Simple meaning                                  |
| ------------------- | ----------------------------------------------- |
| `tenure`            | How long the customer has been with the company |
| `monthly_charges`   | How much they pay each month                    |
| `total_charges`     | How much they have paid overall                 |
| `contract_type`     | Type of contract                                |
| `support_tickets`   | Number of support requests                      |
| `paperless_billing` | Whether paperless billing is enabled            |

These are the clues the model uses to estimate churn.

---

# 4. Let's understand each feature

## `tenure`

This means:

> **How long has the customer been with us?**

Your code generates values from:

```text
1 → 72 months
```

So:

```text
1 month
6 months
24 months
48 months
72 months
```

In your synthetic data:

> **Longer tenure reduces churn probability.**

That's intuitive: a customer who has stayed for years may be less likely to leave than someone who joined recently.

---

# 5. `monthly_charges`

This means:

> **How much does the customer pay every month?**

Your code generates values between:

```text
$20 → $120
```

For example:

```text
Customer A → $30/month
Customer B → $90/month
```

Your synthetic formula treats higher monthly charges as slightly increasing churn risk.

The important word is **slightly**.

---

# 6. `contract_type`

This is an important feature.

Your code generates:

```python
contract_type = np.random.choice(
    [0, 1, 2],
    p=[0.55, 0.25, 0.20]
)
```

So the values are:

```text
0
1
2
```

The original documentation interprets these as:

```text
0 = Month-to-month
1 = 1-year
2 = 2-year
```

The code itself only creates the numbers `0`, `1`, and `2`; the meaning comes from your application's definition.

Based on your formula:

```python
-1.2 * contract_type
```

a higher contract type value reduces the synthetic churn score.

So conceptually:

```text
Month-to-month
     ↓
Higher churn risk

1-year contract
     ↓
Lower churn risk

2-year contract
     ↓
Even lower churn risk
```

This makes sense because customers with longer commitments have more reason to stay.

---

# 7. `support_tickets`

This represents:

> **How many times has the customer contacted support?**

For example:

```text
0 tickets → no recent support problems
1 ticket  → one issue
4 tickets → several issues
```

Your synthetic model assumes:

> **More support tickets → higher churn risk.**

Why?

Because frequent support problems may indicate that the customer is frustrated.

For example:

```text
Customer
   ↓
Problem
   ↓
Contacts support
   ↓
Problem still exists
   ↓
Contacts support again
   ↓
Customer becomes frustrated
   ↓
Higher chance of leaving
```

---

# 8. `paperless_billing`

This is a simple 0/1 feature.

```text
0 → No
1 → Yes
```

Your synthetic formula gives this feature a small positive effect on churn risk.

So in your generated data:

```text
paperless_billing = 1
```

adds some churn risk.

However, this is just a relationship you created in the synthetic data. It should **not** be interpreted as a real-world finding that paperless billing causes churn.

---

# 9. What is `total_charges`?

This represents:

> **How much money the customer has paid in total.**

The code calculates it approximately as:

```python
total_charges = (
    tenure * monthly_charges +
    random_noise
)
```

For example:

```text
Tenure = 12 months
Monthly charge = $50

12 × $50
= $600

Total charges ≈ $600
```

The random noise means the number won't always be exactly the multiplication result.

An important detail:

> **`total_charges` is included as a model feature, but it is NOT directly used in the synthetic churn formula.**

That means the generated churn labels don't directly depend on `total_charges`.

The Gradient Boosting model can still discover correlations involving it because `total_charges` is related to tenure and monthly charges.

---

# 10. Where does the training data come from?

Just like your other two projects, this project uses **synthetic data**.

The code generates:

```python
n_samples = 1500
```

So:

> **1,500 artificial customers are created.**

A simplified dataset might look like:

```text
Tenure | Monthly | Contract | Tickets | Paperless | Churn
----------------------------------------------------------
3      | $80      | 0        | 4       | 1         | YES
36     | $50      | 2        | 0       | 0         | NO
8      | $90      | 0        | 3       | 1         | YES
48     | $40      | 2        | 1       | 0         | NO
```

The model learns from these examples.

---

# 11. How does the program decide who churns?

This is where the code creates a synthetic "churn risk score."

It calculates:

```python
logit = (
    -0.04 * tenure +
    0.02 * monthly_charges +
    -1.2 * contract_type +
    0.5 * support_tickets +
    0.3 * paperless_billing +
    noise
)
```

Don't worry about the mathematical formula.

Think of it as a points system.

```text
Factor                         Effect

Longer tenure                  ↓ churn risk

Higher monthly charge          ↑ churn risk

Longer contract                ↓ churn risk

More support tickets           ↑ churn risk

Paperless billing              ↑ churn risk
```

Then the program adds a small amount of random noise.

---

# 12. What does the sigmoid function do?

After calculating the risk score, the code does:

```python
prob = 1 / (1 + np.exp(-logit))
```

This converts the mathematical score into a probability between 0 and 1.

For example:

```text
Risk calculation
      ↓
Sigmoid function
      ↓
0.85
```

Then we can say:

```text
Churn probability = 85%
```

So the sigmoid is basically a **converter from a mathematical score into a probability-like value**.

---

# 13. How does the program create the churn label?

The code then does:

```python
churn = (
    np.random.uniform(0, 1, size=n_samples) < prob
).astype(int)
```

This introduces randomness.

For example:

```text
Predicted probability = 80%

Random number = 30%

30% < 80%

→ Churn = 1
```

But:

```text
Predicted probability = 20%

Random number = 70%

70% < 20%

→ Churn = 0
```

So:

```text
1 → Customer churned
0 → Customer stayed
```

---

# 14. Why do we split the data?

The code uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y
)
```

That gives us:

```text
1,500 customers
       ↓
 ┌───────────────┐
 │               │
 ↓               ↓
80%             20%
Training        Testing
~1,200          ~300
```

The training customers teach the model.

The testing customers are used to check whether the model can make predictions on customers it hasn't seen during training.

---

# 15. What is Gradient Boosting?

This is your machine-learning algorithm.

Your code uses:

```python
GradientBoostingClassifier(
    n_estimators=120,
    max_depth=4,
    random_state=RANDOM_SEED
)
```

The easiest way to understand Gradient Boosting is:

> **It builds many small decision trees one after another, with later trees trying to improve on the mistakes made by earlier trees.**

Imagine this:

```text
Tree 1
  ↓
Makes some mistakes

Tree 2
  ↓
Focuses on improving those mistakes

Tree 3
  ↓
Improves further

Tree 4
  ↓
Improves further

...

Tree 120
  ↓
Final combined prediction
```

That's the basic idea behind gradient boosting.

---

# 16. How is Gradient Boosting different from your Fraud Detection Random Forest?

This is a great thing to understand because you now have both models.

### Fraud Detection

Uses:

```text
Random Forest
```

Think:

> **Many trees work independently and their results are combined.**

---

### Customer Churn

Uses:

```text
Gradient Boosting
```

Think:

> **Trees are built sequentially, with later trees trying to correct earlier mistakes.**

Very simplified:

```text
Random Forest:

Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤
Tree 4 ─┤──→ Combined prediction
...     │
Tree 100┘


Gradient Boosting:

Tree 1
  ↓
Tree 2 improves mistakes
  ↓
Tree 3 improves mistakes
  ↓
...
  ↓
Tree 120
  ↓
Final prediction
```

That's enough to understand the difference at your current level.

---

# 17. Why `n_estimators=120`?

Your code says:

```python
n_estimators=120
```

That means the Gradient Boosting model uses **120 trees/boosting stages**.

Think:

> **The model gets 120 opportunities to improve its prediction.**

More isn't automatically better. More trees can also increase computation and potentially overfit.

---

# 18. What does `max_depth=4` mean?

Each individual tree can grow to a maximum depth of 4.

Think of each tree asking a small number of questions:

```text
Question 1
   ↓
Question 2
   ↓
Question 3
   ↓
Question 4
```

This keeps the individual trees relatively shallow.

---

# 19. How does the model learn?

This line performs the learning:

```python
model.fit(X_train, y_train)
```

The model sees examples such as:

```text
Customer A:
Tenure = 3
Tickets = 4
Contract = 0
       ↓
Churn = YES


Customer B:
Tenure = 48
Tickets = 0
Contract = 2
       ↓
Churn = NO
```

The model tries to learn patterns that distinguish:

```text
Customers who churn
```

from:

```text
Customers who stay
```

---

# 20. How do we evaluate the model?

After training, the model predicts the test customers:

```python
y_pred = model.predict(X_test)
```

Then your code calculates:

```python
accuracy_score(y_test, y_pred)
```

and:

```python
f1_score(y_test, y_pred)
```

---

# 21. What is accuracy?

Accuracy asks:

> **"Out of all the test customers, how many did we classify correctly?"**

For example:

```text
100 customers tested

72 predictions correct
28 predictions incorrect

Accuracy = 72%
```

So if your documentation says approximately:

```text
Accuracy ≈ 0.72
```

that means roughly 72% of the test predictions were correct.

---

# 22. What is F1 score?

F1 is useful because churn prediction often cares about correctly identifying the customers who actually leave.

It combines:

**Precision**

> When the model says a customer will churn, how often is it right?

and

**Recall**

> Of all the customers who actually churn, how many did the model find?

F1 combines those two into one score.

So if:

```text
F1 ≈ 0.75
```

you can say:

> **The model has a reasonably balanced ability to identify churners while avoiding too many incorrect churn predictions.**

---

# 23. What happens after training?

The model is saved:

```python
self.save()
```

under:

```text
customer_churn_model.joblib
```

The saved model contains:

```text
Gradient Boosting classifier
+
Feature names
+
Evaluation metrics
```

This means the application can later load the trained model instead of training it every time.

---

# 24. What happens when a new customer arrives?

Now imagine a real customer comes in:

```text
Tenure = 6 months
Monthly charges = $89.90
Total charges = $539.40
Contract type = 0
Support tickets = 4
Paperless billing = 1
```

The `predict()` function receives these values.

First, it loads the model if necessary.

Then it arranges the input features in the same order used during training.

Then:

```python
prob = model_cls.predict_proba(df_input)[0, 1]
```

asks:

> **"What is the probability that this customer will churn?"**

Suppose the model says:

```text
0.85
```

That's:

```text
85% churn probability
```

---

# 25. How does the system determine LOW, MEDIUM, or HIGH?

Your code uses:

```python
risk_level = (
    "HIGH" if prob > 0.6
    else "MEDIUM" if prob > 0.3
    else "LOW"
)
```

So:

```text
Probability

0% ───────── 30% ───────── 60% ───────── 100%
       LOW          MEDIUM          HIGH
```

More precisely:

```text
≤ 30%       → LOW
>30%–60%    → MEDIUM
>60%        → HIGH
```

These are **your application's thresholds**.

They aren't automatically learned by Gradient Boosting.

---

# 26. But then why does the code use 50% for retention?

This is an important detail.

Your risk levels use:

```text
30%
60%
```

But your retention strategy uses:

```python
if prob > 0.5:
```

So the two systems are slightly different.

For example:

```text
Churn probability = 55%
```

The customer gets:

```text
MEDIUM
```

because 55% is below 60%.

But:

```text
55% > 50%
```

so they **still receive a retention action**.

Therefore:

> **Risk classification and retention-action eligibility are two separate rules in your code.**

That's worth remembering.

---

# 27. How does the retention system work?

This is the second major part of your project.

The machine-learning model answers:

> **"How likely is this customer to leave?"**

Then your rule engine answers:

> **"What should we do about it?"**

For customers with:

```text
Churn probability > 50%
```

your code looks at additional information.

---

# 28. Rule 1 — Month-to-month customer

If:

```python
contract_type == 0
```

the system recommends:

> **Offer a 15% discount for a 12-month contract lock-in.**

The reasoning is:

```text
Customer is likely to churn
+
Customer is month-to-month
        ↓
Give an incentive to commit
        ↓
Offer discount for longer contract
```

---

# 29. Rule 2 — Lots of support tickets

If:

```python
support_tickets >= 3
```

the system recommends:

> **Assign a dedicated customer success manager to resolve the customer's issues.**

The idea is:

```text
High churn probability
+
Many support problems
        ↓
Customer may be frustrated
        ↓
Give them dedicated support
```

---

# 30. Rule 3 — High churn but neither condition

If the customer has:

```text
Churn probability > 50%
```

but:

```text
contract_type != 0
```

and:

```text
support_tickets < 3
```

the system recommends:

> **Send a personalized appreciation gift and loyalty perk.**

So the logic is:

```text
                 Churn > 50%?
                     │
            ┌────────┴────────┐
           NO                YES
            │                  │
            ↓                  ↓
       Newsletter       Contract type = 0?
                              │
                     ┌────────┴────────┐
                    YES                NO
                     │                  │
                     ↓                  ↓
                  Discount       Tickets >= 3?
                                      │
                              ┌───────┴───────┐
                             YES             NO
                              │               │
                              ↓               ↓
                           Customer       Gift +
                           success        loyalty perk
                           manager
```

---

# 31. What happens to low-risk customers?

If:

```text
churn probability ≤ 50%
```

the code doesn't give them a special retention intervention.

Instead:

```text
Standard nurturing newsletter
+
Upsell campaign
```

So the system doesn't spend expensive retention resources on everyone.

It focuses stronger interventions on customers considered more likely to leave.

---

# 32. Let's walk through your example

Input:

```json
{
  "tenure": 6,
  "monthly_charges": 89.9,
  "total_charges": 539.4,
  "contract_type": 0,
  "support_tickets": 4,
  "paperless_billing": 1
}
```

Suppose the model predicts:

```text
Churn probability = 85%
```

Then:

```text
85% > 60%
```

Therefore:

```text
Risk = HIGH
```

Next:

```text
85% > 50%
```

So the retention engine activates.

Then it checks:

```text
contract_type = 0
```

That's true.

Therefore:

```text
Recommended action:
Offer 15% discount for 12-month contract lock-in
```

So the complete result is conceptually:

```text
Customer
   ↓
85% churn probability
   ↓
HIGH risk
   ↓
Month-to-month customer
   ↓
Offer 15% discount for 12-month contract
```

---

# 33. What does the final API response actually contain?

Your actual `predict()` method returns:

```json
{
  "churn_probability": 0.85,
  "risk_level": "HIGH",
  "recommended_action": "Offer 15% discount for 12-month contract lock-in"
}
```

Notice that the code calls the field:

```text
recommended_action
```

not:

```text
retention_action
```

So the API example in your original documentation should be updated.

---

# 34. The entire project in one diagram

```text
                  👤 CUSTOMER
                      │
                      ↓
          ┌────────────────────────┐
          │ Customer information   │
          │                        │
          │ Tenure                 │
          │ Monthly charges        │
          │ Total charges          │
          │ Contract type          │
          │ Support tickets        │
          │ Paperless billing      │
          └────────────┬───────────┘
                       ↓
              🌳 Gradient Boosting
                       │
                       ↓
               Churn probability
                       │
             Example: 85%
                       │
                       ↓
                Risk classification
                       │
              ┌────────┼────────┐
              ↓        ↓        ↓
            LOW      MEDIUM    HIGH
                       │
                       ↓
             Retention Rule Engine
                       │
             ┌─────────┼─────────┐
             ↓         ↓         ↓
         Discount   Customer    Loyalty
                    Success      Perk
                    Manager
```

---

# 35. The most important distinction

Your project actually contains **two systems**.

### System 1 — Machine Learning

Gradient Boosting answers:

> **"How likely is this customer to churn?"**

Output:

```text
85%
```

### System 2 — Business Rules

Your Python `if/elif` logic answers:

> **"What should we do about this customer?"**

Output:

```text
Offer discount
```

So:

```text
Machine Learning
       ↓
85% churn probability
       ↓
Business Rules
       ↓
Recommended action
```

This distinction is very important in an interview.

The **ML model doesn't decide to give the customer a discount**.

Your **rule-based retention engine** does.

---

# 36. What is actually implemented?

Based on the code you provided:

### ✅ Implemented

* Synthetic customer data generation
* 1,500 customer records
* Six customer features
* Synthetic churn probability generation
* Synthetic churn labels
* 80/20 train-test split
* Stratified split
* Gradient Boosting classifier
* 120 boosting stages
* Maximum tree depth of 4
* Churn probability prediction
* Accuracy calculation
* F1-score calculation
* Model serialization
* LOW/MEDIUM/HIGH risk classification
* Rule-based retention recommendations

### ❌ Not implemented in this code

The original document mentions some things that aren't actually present here:

* Real customer/subscriber data
* CRM integration
* HubSpot/Salesforce/Braze webhooks
* Automatic campaign triggering
* Monthly retraining
* Cron-based retraining
* Real-time customer event streaming
* Customer lifetime value optimization
* A/B testing of retention offers

These could be **future production features**, but they aren't implemented by the code shown.

---

# 37. One thing I would correct in the original documentation

It says:

> "Computes non-linear churn ground-truth score."

That's misleading.

Your synthetic `logit` is actually a **linear combination of the features**:

```text
tenure
+
monthly charges
+
contract type
+
support tickets
+
paperless billing
```

The final probability conversion uses a sigmoid, but there are no explicit feature interactions or nonlinear transformations in the synthetic formula.

The **Gradient Boosting model itself can learn nonlinear relationships**, but the synthetic label-generation formula is basically linear before the sigmoid.

So I'd say:

> **"Computes a synthetic churn risk score from weighted customer features and converts it into a probability."**

That's more accurate.

---

# 38. How I would explain this project in an interview

If someone asks:

### "Tell me about your customer churn project."

Say:

> **"I built a customer churn prediction pipeline using Gradient Boosting. The goal is to identify customers who are likely to cancel or leave a service so the company can intervene before they churn.**
>
> **For the prototype, I generated 1,500 synthetic customer records containing tenure, monthly charges, total charges, contract type, support tickets, and paperless billing. I generated synthetic churn labels based on factors such as shorter tenure, higher charges, shorter contracts, and more support tickets.**
>
> **I split the data into 80% training and 20% testing using stratification, then trained a Gradient Boosting classifier with 120 estimators and a maximum tree depth of 4. I evaluated it using accuracy and F1-score.**
>
> **During inference, the model produces a churn probability. I classify customers as LOW, MEDIUM, or HIGH risk using 30% and 60% thresholds. Then, separately from the ML model, I use business rules to recommend a retention action. For example, a high-churn-probability month-to-month customer receives a discount recommendation, while a high-risk customer with many support tickets is recommended for dedicated customer-success support."**

---

# 39. The simplest possible explanation

If you need to explain it to someone who knows **nothing about machine learning**, say:

> **"Imagine a company has thousands of customers and wants to know who might leave. My system looks at information such as how long someone has been a customer, how much they pay, what type of contract they have, and how many times they've contacted support.**
>
> **A machine-learning model looks at those patterns and estimates the probability that each customer will leave.**
>
> **For example, if a customer has an 85% chance of leaving, the system marks them as high risk. It then checks some business rules and recommends an action, such as offering a discount or assigning a customer-success manager.**
>
> **So the project is essentially: predict who might leave, then decide how to try to keep them."**

# 40. The one line to remember

**Customer data → Gradient Boosting → Churn probability → Risk level → Retention action**

And compared with your other projects:

```text
Fraud Detection
Transaction → Random Forest → Fraud probability → Risk

Credit Risk
Borrower → Logistic Regression → Default probability → Loan decision

Customer Churn
Customer → Gradient Boosting → Churn probability → Retention action
```

Once you understand those three flows, you've understood the core of all three projects.
