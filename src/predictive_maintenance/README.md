# ⚙️ Predictive Maintenance

## Beginner-Friendly Explanation

## 1. What problem are we solving?

Imagine a factory has a machine running 24 hours a day.

The machine has sensors measuring things like:

```text
🌡️ Temperature
📳 Vibration
💨 Pressure
⚙️ RPM
📡 Sensor noise
⏱️ Operating hours
```

The company doesn't want to wait until the machine breaks.

Why?

Because:

```text
Machine breaks
      ↓
Production stops
      ↓
Repair required
      ↓
Money lost
```

Instead, they want to ask:

> **"Does this machine look like it's about to fail?"**

That's **predictive maintenance**.

---

# 2. What is predictive maintenance?

There are three basic approaches to maintaining a machine.

### Approach 1 — Fix it after it breaks

```text
Machine runs
    ↓
Machine breaks
    ↓
Repair it
```

This is called **reactive maintenance**.

Problem:

> You find out about the problem too late.

---

### Approach 2 — Service it on a fixed schedule

For example:

```text
Every 6 months
    ↓
Inspect machine
```

This is **preventive maintenance**.

Better, but sometimes unnecessary.

The machine might be perfectly healthy after 6 months.

---

### Approach 3 — Predict when it needs attention

```text
Sensor readings
      ↓
ML model
      ↓
Failure risk
      ↓
Maintenance decision
```

This is **predictive maintenance**.

That's what your project demonstrates.

---

# 3. What does your system actually receive?

Your API receives:

```json
{
  "vibration_hz": 68.5,
  "temperature_c": 92.3,
  "pressure_psi": 78.0,
  "rpm": 2800,
  "sensor_noise_std": 3.2,
  "operating_hours": 6500
}
```

Don't think of these as random numbers.

Think:

> **These are the machine's current health measurements.**

For example:

```text
Vibration → 68.5 Hz
Temperature → 92.3 °C
Pressure → 78 PSI
RPM → 2800
Noise → 3.2
Operating hours → 6500
```

The model looks at all of them together.

---

# 4. Why do we need multiple sensors?

Imagine you only monitored temperature.

You could create a simple rule:

```text
Temperature > 100°C
        ↓
Danger
```

But machines don't always fail because of one measurement.

You might have:

```text
Temperature → moderately high
Vibration   → high
Pressure    → moderately high
Operating hours → very high
```

Individually, none might look terrible.

But **together**, they might indicate a problem.

That's where machine learning becomes useful.

---

# 5. What data does your model use?

Your six inputs are:

| Feature            | Simple meaning                          |
| ------------------ | --------------------------------------- |
| `vibration_hz`     | How much the machine is vibrating       |
| `temperature_c`    | Machine temperature                     |
| `pressure_psi`     | Operating pressure                      |
| `rpm`              | Rotational speed                        |
| `sensor_noise_std` | How noisy/unstable the sensor signal is |
| `operating_hours`  | How long the machine has been running   |

The model takes these six values and tries to answer:

> **"Is this machine likely to fail?"**

---

# 6. But where does the training data come from?

This is very important.

Your code does **not** use real factory data.

It creates:

```python
n_samples = 1500
```

synthetic machine records.

So you're basically creating 1,500 imaginary machine situations.

For example:

```text
Machine 1:
Vibration = 30
Temperature = 65
Pressure = 45
RPM = 1800
Noise = 1
Hours = 1000

Machine 2:
Vibration = 80
Temperature = 100
Pressure = 90
RPM = 3200
Noise = 4
Hours = 9000
```

Then your code decides whether each situation counts as a failure.

---

# 7. How does your code decide what a "failure" is?

This is the most important part of `generate_data()`.

Your code creates:

```python
failure_score = (
    0.05 * (vibration_hz - 40) +
    0.08 * (temperature_c - 70) +
    0.04 * (pressure_psi - 50) +
    0.8 * sensor_noise_std +
    0.0003 * operating_hours +
    random_noise
)
```

Don't worry about the formula.

Think of it as a **risk score calculator**.

Each sensor contributes some amount of risk.

```text
Vibration       → adds risk
Temperature     → adds risk
Pressure        → adds risk
Sensor noise    → adds risk
Operating hours → adds risk
```

Then:

```python
is_failure = (failure_score > 3.2).astype(int)
```

means:

```text
Failure score > 3.2
        ↓
Failure = 1

Otherwise
        ↓
Failure = 0
```

So your synthetic dataset looks roughly like:

```text
Machine data                     Label

Temperature 65
Vibration 30
Hours 1000                      → 0 = No failure

Temperature 100
Vibration 80
Hours 9000                      → 1 = Failure
```

---

# 8. Important: the model is learning from labels created by your formula

This is a very important thing to understand.

You aren't giving the model real historical failure records.

You create the labels yourself:

```text
Sensor values
     ↓
Your formula
     ↓
failure_score
     ↓
0 or 1
```

Then you train Random Forest to learn that relationship.

So this is a **prototype/demo**, not a real industrial predictive-maintenance model.

In a real system, you'd ideally have:

```text
Real sensor history
       +
Real maintenance records
       +
Actual machine failures
       ↓
Training dataset
```

---

# 9. What does `is_failure = 1` mean?

Very simply:

```text
0 → Machine did not fail
1 → Machine failed
```

So this is a **classification problem**.

You're asking the model:

> "Class 0 or Class 1?"

Just like your fraud project:

```text
Fraud:
0 → Not fraud
1 → Fraud
```

Predictive maintenance:

```text
Failure:
0 → Healthy/no failure
1 → Failure
```

---

# 10. Now we train Random Forest

Your model is:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=7
)
```

A Random Forest is basically:

> **Many decision trees working together.**

Imagine one tree asking:

```text
Is temperature high?
       ↓
      Yes
       ↓
Is vibration high?
       ↓
      Yes
       ↓
Are operating hours high?
       ↓
      Yes
       ↓
High failure risk
```

Another tree might look at different combinations.

You create:

```text
100 decision trees
```

and combine their answers.

That's why it's called:

> **Random Forest**

A forest = many trees.

---

# 11. Why not just use one rule?

You could write:

```python
if temperature > 100:
    failure
```

But that is too simplistic.

Your machine might have:

```text
Temperature = 90
Vibration = 85
Operating hours = 9500
```

Maybe that's dangerous even though:

```text
Temperature < 100
```

A Random Forest can learn combinations of conditions.

That's one of the reasons you chose it.

---

# 12. What happens during training?

The training process is:

```text
1500 machine records
        ↓
Separate inputs and labels
        ↓
Split into training and testing data
        ↓
Train Random Forest
        ↓
Evaluate it
        ↓
Save model
```

---

# 13. Why split the data?

Your code uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y
)
```

You have 1,500 examples.

Approximately:

```text
80% → Training
20% → Testing
```

So:

```text
1500 machines
     ↓
1200 training examples
300 testing examples
```

The model learns from the training examples.

Then we test whether it works on examples it hasn't seen before.

---

# 14. What does `stratify=y` do?

This is a small but useful detail.

Suppose your dataset contains:

```text
1200 healthy
300 failures
```

When splitting the data, you want both training and testing sets to have roughly the same proportion.

So `stratify=y` helps preserve the class distribution.

In simple words:

> **"Keep the healthy/failure ratio similar in both datasets."**

---

# 15. What does the model learn?

After training, Random Forest has learned relationships between:

```text
Vibration
Temperature
Pressure
RPM
Sensor noise
Operating hours
```

and:

```text
Failure / No Failure
```

So when a new machine arrives:

```text
Temperature = 92
Vibration = 68
Pressure = 78
...
```

the model can estimate:

```text
Probability of failure = 0.78
```

---

# 16. What does `predict_proba()` mean?

Your code does:

```python
prob = float(
    model_cls.predict_proba(df_input)[0, 1]
)
```

This asks the model:

> **"What probability do you assign to the failure class?"**

For example:

```text
0.10 → 10% failure probability
0.50 → 50%
0.78 → 78%
0.95 → 95%
```

So if the model returns:

```text
0.78
```

you can say:

> **"The model estimates a relatively high probability of failure."**

Be careful: it is a **model probability**, not proof that the machine will fail.

---

# 17. What happens after we get the probability?

Your code has three levels.

### Less than 35%

```python
prob < 0.35
```

→

```text
HEALTHY
```

---

### 35% to 65%

```python
0.35 <= prob < 0.65
```

→

```text
WARNING
```

---

### 65% or higher

```python
prob >= 0.65
```

→

```text
CRITICAL
```

So:

```text
Failure probability

0% ───────── 35% ───────── 65% ───────── 100%
     HEALTHY       WARNING       CRITICAL
```

---

# 18. Why have different statuses?

Because businesses don't want every small risk to trigger an emergency.

Imagine:

```text
Failure probability = 10%
```

You probably don't want:

> "Shut down the factory!"

Instead:

```text
HEALTHY
```

But if:

```text
Failure probability = 80%
```

you might want immediate attention.

So your system turns the probability into an operational decision.

---

# 19. What is the recommendation?

Your code says:

### Healthy

```text
System operating within optimal parameters.
Next routine check standard schedule.
```

Meaning:

> Nothing unusual. Continue normal maintenance.

---

### Warning

```text
Schedule preventative maintenance within 48 hours.
```

Meaning:

> Something looks concerning. Inspect the machine soon.

---

### Critical

```text
Immediate shutdown required.
Schedule technician for component overhaul.
```

Meaning:

> The predicted failure risk is high enough that the system recommends immediate intervention.

---

# 20. So your entire prediction system is:

```text
Machine sensors
      ↓
Temperature
Vibration
Pressure
RPM
Noise
Operating hours
      ↓
Random Forest
      ↓
Failure probability
      ↓
┌─────────────┬──────────────┬──────────────┐
│ < 35%       │ 35–65%       │ >= 65%       │
│             │              │              │
│ HEALTHY     │ WARNING      │ CRITICAL     │
│             │              │              │
│ Normal      │ Inspect soon │ Act now      │
└─────────────┴──────────────┴──────────────┘
```

That's your project.

---

# 21. Now let's talk about RUL

Your output also contains:

```python
est_rul_hours
```

RUL means:

> **Remaining Useful Life**

In plain English:

> **"Approximately how many hours of useful operation might remain?"**

For example:

```text
RUL = 800 hours
```

could mean:

> The system estimates the machine has roughly 800 hours before serious failure risk.

---

# 22. But here's an important correction about your code

Your code calculates RUL like this:

```python
est_rul_hours = int(
    round(
        max(10, (1.0 - prob) * 1200)
    )
)
```

So it is simply:

```text
RUL = (1 - failure probability) × 1200
```

with a minimum of 10 hours.

For example:

### Probability = 0.10

```text
(1 - 0.10) × 1200
= 1080 hours
```

### Probability = 0.50

```text
(1 - 0.50) × 1200
= 600 hours
```

### Probability = 0.90

```text
(1 - 0.90) × 1200
= 120 hours
```

So:

```text
Higher failure probability
        ↓
Lower estimated RUL
```

---

# 23. But this is NOT a true RUL model

This is very important for your interview.

Your code does **not actually predict remaining useful life from historical degradation data**.

It converts:

```text
Failure probability
```

into:

```text
Estimated hours
```

using a fixed formula.

So don't say:

> "The model predicts RUL."

A more accurate statement is:

> **"The current prototype derives a simple RUL estimate from the predicted failure probability."**

A true RUL system would usually need historical machine degradation trajectories and actual failure times.

For example:

```text
Machine age
     ↓
Sensor readings over time
     ↓
Degradation pattern
     ↓
Historical failure
     ↓
Learn how long machines typically survive
     ↓
Predict remaining life
```

Your current project doesn't do that.

---

# 24. Another important correction: the RPM feature

Your generated data includes:

```python
rpm
```

But look at your `failure_score`:

```python
failure_score = (
    vibration
    + temperature
    + pressure
    + sensor_noise
    + operating_hours
    + noise
)
```

Notice:

> **RPM is not included.**

So the synthetic failure labels don't directly depend on RPM.

The Random Forest may still find some accidental relationship involving RPM because the data is random, but there is no intentional RPM → failure relationship in the data-generation formula.

Therefore, don't say:

> "High RPM is one of the synthetic failure triggers."

Your code doesn't actually establish that.

---

# 25. Another important correction: the "failure trigger zones"

Your documentation has a table like:

```text
Vibration > 75
Temperature > 95
Pressure > 85
RPM > 3000
...
```

But those aren't actual hard-coded thresholds in your code.

You never have:

```python
if vibration > 75:
```

or:

```python
if temperature > 95:
```

Instead, your formula continuously combines the variables.

So the table should be described as:

> **"Illustrative risk zones"**

rather than:

> **"Actual failure thresholds used by the model."**

---

# 26. Why is Recall important?

Your training code calculates:

```python
recall = recall_score(y_test, y_pred)
```

Why recall?

Because in predictive maintenance, missing a real failure can be expensive.

Imagine 100 machines are actually going to fail.

Your model catches:

```text
93 machines
```

but misses:

```text
7 machines
```

That's a recall of:

```text
93 / 100 = 93%
```

So recall answers:

> **"Of all the machines that actually failed, how many did my model successfully identify?"**

---

# 27. Why might we prefer high recall?

Imagine two mistakes.

### False positive

Model says:

```text
Machine might fail
```

but it doesn't.

Result:

```text
Technician inspects machine unnecessarily.
```

Cost:

> Some wasted maintenance time.

---

### False negative

Model says:

```text
Machine is healthy
```

but it actually fails.

Result:

```text
Machine breaks
Production stops
Emergency repair
Potential damage
```

Potential cost:

> Much larger.

Therefore, you often care strongly about recall.

---

# 28. What is ROC-AUC?

Your code also calculates:

```python
roc_auc_score()
```

You don't need to memorize the mathematics.

At a beginner level:

> **ROC-AUC measures how well the model separates machines that fail from machines that don't.**

A rough interpretation:

```text
0.5 → barely better than random
0.7 → reasonable
0.8 → good
0.9+ → strong separation
```

If your model gets around:

```text
0.93
```

on this synthetic dataset, that's strong.

But remember:

> **It's strong on your generated data.**

That does not mean it would automatically achieve 0.93 on real factory data.

---

# 29. Why can your synthetic model perform so well?

Because you created the failure labels using a formula based on the same sensor variables you're giving to the model.

You essentially did:

```text
Sensor values
    ↓
Known formula
    ↓
Failure label
```

Then:

```text
Same sensor values
    ↓
Random Forest
    ↓
Try to learn the formula
```

So the problem is relatively easy for the model.

Real industrial data is much messier.

---

# 30. What would real predictive-maintenance data look like?

Instead of generating random values, you'd ideally have:

```text
Machine ID
Timestamp
Vibration
Temperature
Pressure
RPM
Current
Oil quality
Operating hours
Maintenance history
Failure event
```

For example:

```text
Machine 101

08:00 → vibration 42
09:00 → vibration 45
10:00 → vibration 48
11:00 → vibration 54
12:00 → vibration 61
13:00 → vibration 70
14:00 → FAILURE
```

Now the model can learn:

> **How machines behave as they approach failure.**

That's much closer to a real predictive-maintenance system.

---

# 31. Your project is therefore best described as a prototype

I would describe it as:

> **"A prototype predictive-maintenance classifier using synthetic sensor telemetry."**

That's honest and technically accurate.

Don't describe it as:

> "A production-ready industrial failure prediction system."

because the current data and RUL logic aren't production-grade.

---

# 32. What happens when the model is saved?

Your code stores:

```python
self.model = {
    "classifier": model,
    "feature_names": list(X.columns),
    "metrics": ...
}
```

Then:

```python
self.save()
```

So the trained Random Forest can be saved and reused later.

When a prediction request arrives:

```python
if self.model is None:
    self.load()
```

So if the model isn't already in memory, it loads the saved model.

---

# 33. Why save the model?

You don't want to retrain every time someone asks:

> "What's the health of this machine?"

That would be inefficient.

Instead:

```text
Train once
   ↓
Save model
   ↓
Load model
   ↓
Predict many times
```

This is common in ML applications.

---

# 34. A complete example

Suppose your machine reports:

```text
Vibration       = 68.5 Hz
Temperature     = 92.3°C
Pressure        = 78 PSI
RPM             = 2800
Sensor noise    = 3.2
Operating hours = 6500
```

The model processes those values.

Suppose it returns:

```text
Failure probability = 0.78
```

Then your rules say:

```text
0.78 >= 0.65
```

Therefore:

```text
Health status = CRITICAL
```

Then RUL:

```text
(1 - 0.78) × 1200
= 264 hours
```

So the response might conceptually be:

```text
Failure probability → 78%
Health status       → CRITICAL
Estimated RUL       → 264 hours
Action              → Immediate intervention
```

---

# 35. The entire project in one picture

```text
                 MACHINE
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   Temperature   Vibration   Pressure
        ↓           ↓           ↓
       RPM       Sensor Noise   Hours
        └───────────┼───────────┘
                    ↓
              Random Forest
                    ↓
           Failure Probability
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      <35%        35–65%        ≥65%
        ↓           ↓             ↓
    HEALTHY      WARNING       CRITICAL
        ↓           ↓             ↓
   Normal check  Inspect soon   Act now
                    │
                    ↓
             Simple RUL estimate
```

---

# 36. How I would explain the project in an interview

If someone says:

### "Tell me about your predictive maintenance project."

You can say:

> **"I built a predictive-maintenance prototype that uses machine sensor readings to estimate the probability of equipment failure.**
>
> **The input features are vibration, temperature, pressure, RPM, sensor noise, and operating hours. For the prototype, I generated 1,500 synthetic machine records and created failure labels using a risk-scoring formula.**
>
> **I trained a Random Forest classifier because it can capture interactions between multiple sensor measurements instead of relying on individual hardcoded thresholds. I evaluated it using ROC-AUC and recall, with recall being particularly important because missing an actual machine failure can be much more costly than performing an unnecessary inspection.**
>
> **During prediction, the model outputs a failure probability. I then map that probability into three operational states: HEALTHY, WARNING, or CRITICAL. The system also derives a simple RUL estimate from the failure probability and provides a recommended maintenance action."**

---

# 37. If they ask: "Why Random Forest?"

Say:

> **"Because machine failure can depend on combinations of sensor readings. Random Forest can learn nonlinear relationships and interactions between vibration, temperature, pressure, noise, and operating hours without requiring me to manually create every threshold combination."**

---

# 38. If they ask: "Why recall?"

Say:

> **"Because false negatives can be very expensive in predictive maintenance. If the model says a machine is healthy when it's actually about to fail, we may miss the opportunity to intervene before a breakdown. So I prioritize catching as many actual failures as possible."**

---

# 39. If they ask: "What is RUL?"

Say:

> **"RUL means Remaining Useful Life. It estimates how much useful operating time may remain before a machine needs major attention. In my current prototype, the RUL is derived from the failure probability rather than being learned as a separate time-to-failure model."**

That last sentence is especially important because it shows you understand the limitation of your own implementation.

---

# 40. If they ask: "Is this production-ready?"

A strong answer is:

> **"The architecture demonstrates the prediction workflow, but the current implementation is a prototype because it uses synthetic data and a simplified RUL calculation. For production, I would train on real historical sensor telemetry and maintenance/failure records, account for time-dependent degradation, calibrate the failure probabilities, validate the model across different machines, and build a proper time-to-failure or survival model for RUL."**

That answer is much stronger than pretending the prototype is production-ready.

---

# 41. The one sentence to remember

> **Machine sensors → Random Forest → Probability of failure → Health status → Maintenance action**

And now your seven projects can be remembered very simply:

```text
🛡️ Fraud
Transaction → "Is this fraud?"

💳 Credit Risk
Borrower → "Will they default?"

📉 Churn
Customer → "Will they leave?"

🏡 House Price
House → "How much is it worth?"

🎯 Recommendation
User → "What will they like?"

📊 Demand Forecasting
Past sales → "How much will we sell next?"

⚙️ Predictive Maintenance
Machine sensors → "Will it fail?"
```

That's the common pattern across your projects:

**Input data → ML model → prediction → business decision.**
