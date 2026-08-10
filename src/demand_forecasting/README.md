# 📊 Demand Forecasting

## Beginner-Friendly Explanation

## 1. What is this project?

Imagine you own a store.

Every day, you sell some number of products:

```text
Monday    → 105 units
Tuesday   → 112 units
Wednesday → 98 units
Thursday  → 120 units
Friday    → 135 units
Saturday  → 160 units
Sunday    → 150 units
```

Now you ask:

> **"How many products will I probably sell next week?"**

That's **demand forecasting**.

The purpose is to help a business decide:

* How much inventory to keep
* How much stock to order
* Whether there could be shortages
* How much warehouse space is needed
* How much demand a promotion might create

So the basic idea is:

```text
Past sales
    +
Calendar information
    +
Promotion information
    ↓
Machine-learning model
    ↓
Future sales prediction
```

---

# 2. What makes this different from House Price Prediction?

Your House Price project asks:

> "Given this house, what is its price?"

Each house is basically an independent example.

But Demand Forecasting asks:

> **"Given what happened before, what will happen next?"**

The order of the data matters.

For example:

```text
Monday → 100
Tuesday → 105
Wednesday → 110
Thursday → ?
```

The previous values can help predict Thursday.

So this is a **time-series forecasting problem**.

---

# 3. Your system generates 500 days of data

Your code starts with:

```python
def generate_data(self, n_days: int = 500):
```

So it creates:

> **500 days of artificial sales history.**

The first date is:

```python
start_date = datetime(2025, 1, 1)
```

Then it creates:

```text
2025-01-01
2025-01-02
2025-01-03
...
```

until it has 500 days.

---

# 4. What information does each day contain?

For every day, your program creates:

```text
date
day_of_week
month
is_weekend
is_promo
day_of_year
units_sold
```

Later, it also creates:

```text
lag_1
lag_7
rolling_mean_7
```

So eventually the model has several clues about each day.

---

# 5. Let's understand the calendar features

## `day_of_week`

Your code does:

```python
day_of_week = d.weekday()
```

Python represents the days as:

```text
Monday    → 0
Tuesday   → 1
Wednesday → 2
Thursday  → 3
Friday    → 4
Saturday  → 5
Sunday    → 6
```

Why is this useful?

Because stores often sell differently on different days.

For example:

```text
Monday → 100 units
Tuesday → 105 units
Friday → 130 units
Saturday → 160 units
```

The model needs to know what day it is.

---

# 6. `is_weekend`

Your code says:

```python
is_weekend = 1 if day_of_week >= 5 else 0
```

So:

```text
Monday → 0
Tuesday → 0
Wednesday → 0
Thursday → 0
Friday → 0
Saturday → 1
Sunday → 1
```

This gives the model an easy signal:

> **"Is today a weekend?"**

Your synthetic data assumes weekends have higher demand.

---

# 7. `month`

This is simply:

```text
January   → 1
February  → 2
March     → 3
...
December  → 12
```

The idea is that demand can change during different parts of the year.

For example, a real business might have:

```text
December → high demand
January  → low demand
Summer   → high demand
```

Your model can use the month as one of its clues.

---

# 8. `day_of_year`

This is the number of the day within the year.

For example:

```text
January 1 → 1
January 2 → 2
...
December → around 365
```

Your code uses this for the seasonal pattern.

---

# 9. `is_promo`

This is very important.

Your code randomly decides whether a promotion is running:

```python
is_promo = np.random.choice(
    [0, 1],
    p=[0.85, 0.15]
)
```

This means:

```text
85% → No promotion
15% → Promotion
```

So roughly 15% of the days have a promotion.

The synthetic data assumes:

> **Promotion → more products sold.**

---

# 10. Now we create the actual sales number

This is the most important part of `generate_data()`.

Your code creates:

```python
base_demand = (
    100 +
    15 * sin(...) +
    25 * is_weekend +
    40 * is_promo +
    0.05 * days_since_start +
    random_noise
)
```

Don't worry about the mathematical notation.

Think of it as:

```text
Base demand
    +
Seasonal effect
    +
Weekend effect
    +
Promotion effect
    +
Long-term growth
    +
Random variation
    =
Today's sales
```

---

# 11. Let's break that down

The starting point is:

```python
100
```

So the basic demand is around:

> **100 units per day.**

Then several things modify that number.

---

# 12. Seasonal effect

Your code has:

```python
15 * np.sin(...)
```

This creates a smooth yearly cycle.

Imagine demand going up and down during the year:

```text
Demand
  ↑
  │       /‾‾\
  │      /    \
  │_____/      \____
  │
  └──────────────────→ Time
       One year
```

So demand isn't exactly the same every month.

This is called **seasonality**.

---

# 13. Weekend effect

Your code adds:

```python
25 * is_weekend
```

If it's a weekday:

```text
is_weekend = 0

25 × 0 = 0
```

No extra demand.

If it's Saturday or Sunday:

```text
is_weekend = 1

25 × 1 = +25
```

So weekends get approximately:

> **25 additional units of demand.**

For example:

```text
Weekday:
100 + other effects

Weekend:
100 + 25 + other effects
```

---

# 14. Promotion effect

Your code adds:

```python
40 * is_promo
```

No promotion:

```text
40 × 0 = 0
```

Promotion:

```text
40 × 1 = +40
```

So a promotion adds approximately:

> **40 units of demand**

in the synthetic data.

---

# 15. Long-term trend

Your code has:

```python
0.05 * (d - start_date).days
```

This means demand gradually increases as time passes.

For example:

```text
Day 0   → +0
Day 100 → +5
Day 200 → +10
Day 400 → +20
```

So the synthetic business has a small upward growth trend.

Conceptually:

```text
Sales
  ↑
  │             /
  │           /
  │         /
  │       /
  │_____/
  └────────────────→ Time
```

---

# 16. Random noise

Finally:

```python
np.random.normal(0, 10)
```

adds randomness.

Why?

Because real sales aren't perfectly predictable.

Even if today and tomorrow are both Saturdays with promotions, sales won't necessarily be identical.

You might have:

```text
Saturday → 160
Next Saturday → 153
Next Saturday → 168
```

Random variation makes the synthetic data less perfect.

---

# 17. So how is `units_sold` created?

Your program basically says:

```text
100
+
Seasonality
+
Weekend effect
+
Promotion effect
+
Long-term growth
+
Random noise
=
Units sold
```

For example, a day might end up with:

```text
units_sold = 154
```

Another day:

```text
units_sold = 107
```

Another:

```text
units_sold = 181
```

---

# 18. Why does the code use `max(10, ...)`?

You have:

```python
units_sold = max(10, int(round(base_demand)))
```

This means:

> **Never allow generated demand to go below 10 units.**

If the formula produces:

```text
7 units
```

the system changes it to:

```text
10 units
```

---

# 19. Now comes the really important part: Lag Features

This is what makes your project a forecasting model.

Your code creates:

```python
lag_1
lag_7
rolling_mean_7
```

These features basically tell the model:

> **"What happened recently?"**

---

# 20. What is `lag_1`?

`lag_1` means:

> **Yesterday's sales.**

Suppose:

```text
Monday    → 100
Tuesday   → 110
Wednesday → 120
```

For Wednesday:

```text
lag_1 = Tuesday's sales
      = 110
```

So:

```text
Today's prediction
       ↑
Yesterday's sales
```

The model can use yesterday's demand as a clue.

---

# 21. What is `lag_7`?

`lag_7` means:

> **Sales from 7 days ago.**

Suppose today is Wednesday.

Then `lag_7` is:

> Last Wednesday's sales.

Why is that useful?

Because people often have weekly shopping patterns.

For example:

```text
This Saturday
       ↑
Last Saturday
```

is often more useful than:

```text
This Saturday
       ↑
Yesterday's Friday
```

for understanding weekly seasonality.

So:

```text
lag_1 → yesterday
lag_7 → same day last week
```

---

# 22. What is `rolling_mean_7`?

This means:

> **The average sales over the previous 7 days.**

Suppose the previous seven days had:

```text
100
110
105
120
115
125
105
```

The average is approximately:

```text
111.4
```

So:

```text
rolling_mean_7 ≈ 111.4
```

This gives the model a smoother picture of recent demand.

Instead of looking at one noisy day, it sees:

> **"What has demand been like recently on average?"**

---

# 23. Why use all three?

Because each gives different information.

```text
lag_1
 ↓
Very recent demand

lag_7
 ↓
Weekly pattern

rolling_mean_7
 ↓
Recent overall trend
```

Together:

```text
Recent demand
+
Weekly demand
+
Recent average
```

give the model useful information about the future.

---

# 24. Now we train the model

Your model is:

```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4
)
```

This is similar to your House Price project.

The difference is the input.

### House Price

```text
House characteristics
        ↓
Predict price
```

### Demand Forecasting

```text
Time + sales history + promotion
        ↓
Predict future sales
```

---

# 25. Why Gradient Boosting?

Gradient Boosting builds many decision trees sequentially.

Think:

```text
Tree 1
   ↓
Makes prediction

Tree 2
   ↓
Corrects mistakes

Tree 3
   ↓
Corrects remaining mistakes

...

Tree 100
   ↓
Final prediction
```

It can learn relationships such as:

```text
Weekend + Promotion
        ↓
Very high demand
```

or:

```text
Weekday + No promotion
        ↓
Lower demand
```

without you manually writing every rule.

---

# 26. Why is this called autoregressive?

Because the model uses previous values of the thing it's trying to predict.

Here:

```text
Thing we're predicting:
units_sold
```

And the model uses:

```text
previous units_sold
```

through:

```text
lag_1
lag_7
rolling_mean_7
```

That's why the original documentation calls it:

> **Lag-feature / autoregressive forecasting.**

You don't need to memorize the fancy term.

Just remember:

> **The model uses previous sales to help predict future sales.**

---

# 27. Why doesn't the code shuffle the data?

This is extremely important.

You use:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

Normally, machine-learning datasets are often shuffled.

But time-series data should usually preserve chronological order.

Imagine:

```text
January
February
March
...
October
November
December
```

You want:

```text
TRAIN:
January → October

TEST:
November → December
```

Not:

```text
TRAIN:
January, March, July, December, May...

TEST:
February, April, June...
```

Why?

Because in real life:

> **You train using the past and predict the future.**

You shouldn't let future observations leak into training.

---

# 28. Your train/test split

You have 500 days.

80%:

```text
500 × 0.8 = 400 days
```

20%:

```text
500 × 0.2 = 100 days
```

So approximately:

```text
2025
──────────────────────────────────────────────→

|---------- TRAIN ----------|---- TEST ----|
       ~400 days                ~100 days
```

That's much more appropriate for a forecasting problem than random shuffling.

---

# 29. What is MAE?

Your evaluation uses:

```python
mean_absolute_error()
```

MAE means:

> **Mean Absolute Error**

It basically answers:

> **"On average, how many units away are my predictions from the actual sales?"**

Suppose actual sales are:

```text
100
120
150
```

and predictions are:

```text
110
115
140
```

The errors are:

```text
10
5
10
```

The average absolute error is:

```text
(10 + 5 + 10) / 3
= 8.33 units
```

So:

> **MAE is easy to understand because it's expressed in the same unit you're predicting.**

If:

```text
MAE = 14.49
```

you can roughly say:

> **"The model's daily predictions are off by about 14.5 units on average."**

---

# 30. What is R² here?

You also calculate:

```python
r2_score()
```

R² measures how much variation in sales the model explains.

Your documentation says approximately:

```text
R² ≈ 0.27
```

That means the model explains roughly:

> **27% of the variation in the test-set demand.**

This isn't necessarily terrible for a simple synthetic forecasting setup, but it's much weaker than your house-price model's R².

And that's understandable because demand is noisy and time-dependent.

---

# 31. Why is the R² only around 0.27?

Your data contains:

```text
Seasonality
+
Weekend effect
+
Promotion
+
Trend
+
Random noise
```

The model has access to many of those signals.

But it can't perfectly predict:

```text
Random noise
```

And there is another important issue:

> Your synthetic data has a smooth yearly seasonal pattern, but the feature representation is fairly basic.

You provide:

```text
month
day_of_year
```

but not explicit sine/cosine seasonal features.

So the model has to approximate the seasonal curve using trees.

Also, your forecasting process at inference is simplified, as we'll see later.

---

# 32. Now let's understand prediction

Suppose the API receives:

```json
{
  "store_id": "STORE_101",
  "horizon_days": 7,
  "is_promo": 1
}
```

This means:

> "Forecast demand for the next 7 days, assuming there is a promotion."

The system creates:

```text
Day 1 → prediction
Day 2 → prediction
Day 3 → prediction
...
Day 7 → prediction
```

---

# 33. What does `horizon_days` mean?

If:

```text
horizon_days = 7
```

the system predicts:

> **The next 7 days.**

If:

```text
horizon_days = 30
```

it predicts:

> **The next 30 days.**

So:

```text
horizon_days
     ↓
How far into the future should we forecast?
```

---

# 34. What is `store_id`?

The input contains:

```python
store_id = input_data.get("store_id", "STORE_101")
```

This allows the API to identify the store.

But here's an important detail:

> **Your model does not actually use `store_id` as a prediction feature.**

The value is simply returned in the output.

So if you send:

```text
STORE_101
```

or:

```text
STORE_999
```

the model makes the prediction using the same learned model.

There is no store-specific behavior in the current code.

This means your current pipeline is really:

> **A single generic demand model, with `store_id` acting as metadata.**

---

# 35. How does the model predict Day 1?

The code starts with:

```python
running_lag1 = self.model["last_units_mean"]
running_lag7 = self.model["last_units_mean"]
running_roll7 = self.model["last_units_mean"]
```

This means the system doesn't actually have the latest real sales history available at prediction time.

Instead, it initializes the lag values using:

> **The average units sold in the training dataset.**

So the initial state is approximately:

```text
lag_1 = average historical demand
lag_7 = average historical demand
rolling_mean_7 = average historical demand
```

Then it predicts the first future day.

---

# 36. This is important: the current inference isn't using real recent sales

In a real forecasting system, you'd ideally know:

```text
Yesterday's actual sales
Sales 7 days ago
Last 7 days of sales
```

Then you'd initialize:

```text
lag_1
lag_7
rolling_mean_7
```

using those actual values.

Your current code instead uses:

```text
Overall historical mean
```

for all three starting values.

That's okay for a prototype, but it limits forecast quality.

---

# 37. What does recursive forecasting mean?

This is one of the most important ideas in your project.

Suppose we want to predict:

```text
Tomorrow
Day 2
Day 3
Day 4
```

For tomorrow, we don't know tomorrow's actual sales.

So:

```text
Predict Day 1
     ↓
Use Day 1 prediction as information
     ↓
Predict Day 2
     ↓
Use Day 2 prediction
     ↓
Predict Day 3
```

That's called **recursive forecasting**.

---

# 38. Your code does exactly this

After predicting a day:

```python
running_lag1 = pred_units
```

So if:

```text
Day 1 prediction = 130
```

then for the next day:

```text
lag_1 = 130
```

The model treats its own prediction as yesterday's sales.

Then it predicts Day 2.

This continues:

```text
Prediction Day 1
       ↓
becomes lag_1
       ↓
Prediction Day 2
       ↓
becomes lag_1
       ↓
Prediction Day 3
       ↓
...
```

---

# 39. What happens to the rolling mean?

Your code does:

```python
running_roll7 = (
    running_roll7 * 6 + pred_units
) / 7
```

Conceptually, it's updating the recent average using the new prediction.

So:

```text
Old rolling average
       +
New predicted demand
       ↓
Updated rolling average
```

Again, the future forecast is gradually feeding into later forecasts.

---

# 40. What happens to `lag_7`?

Here's a subtle issue in your code.

You initialize:

```python
running_lag7 = self.model["last_units_mean"]
```

but inside the loop you never update:

```python
running_lag7
```

So the value of `lag_7` remains the same for every forecasted day.

That's important.

The documentation says:

> "updates autoregressive lag buffers dynamically"

But that's only partly true.

`lag_1` is updated.

`rolling_mean_7` is updated.

**`lag_7` is not updated.**

So the actual implementation is simpler than the documentation suggests.

---

# 41. What does a more complete recursive forecast normally do?

Ideally, you would maintain a history:

```text
Day -6
Day -5
Day -4
Day -3
Day -2
Day -1
Day 0
```

Then:

```text
Forecast Day 1
    ↓
Add it to history

Forecast Day 2
    ↓
Use the value from 7 days ago

Forecast Day 3
    ↓
Use the updated 7-day history
```

That would allow:

```text
lag_7
```

to actually represent:

> **The predicted/actual sales from seven days earlier.**

Your current implementation doesn't do that.

---

# 42. What does the final forecast contain?

For every day, your code returns:

```text
date
day_name
forecasted_units
lower_bound
upper_bound
```

For example:

```text
Monday
Forecast = 120
Lower = 105.6
Upper = 134.4
```

---

# 43. What are the lower and upper bounds?

Your code does:

```python
lower_bound = pred_units * 0.88
upper_bound = pred_units * 1.12
```

So it's simply:

```text
Prediction - 12%
Prediction + 12%
```

For example:

```text
Forecast = 100

Lower = 88
Upper = 112
```

So:

```text
88 ───────── 100 ───────── 112
lower        forecast      upper
```

---

# 44. Important correction: these are NOT confidence intervals

This is another thing I would change in your documentation.

The code does not calculate a statistical confidence interval.

It simply applies:

```text
±12%
```

to the prediction.

So don't call it:

> "95% confidence interval"

or:

> "statistical prediction interval."

A better description is:

> **"A simple ±12% planning range around the forecast."**

That's exactly what your code is doing.

---

# 45. What is `total_forecast_units`?

Suppose the next 7 days are predicted as:

```text
120
135
142
110
95
205
210
```

The system adds them:

```text
120 + 135 + 142 + 110 + 95 + 205 + 210
```

and returns:

```text
Total predicted units
```

So the business can answer:

> **"How much stock might we need for the next 7 days?"**

---

# 46. The complete prediction flow

Your system basically does:

```text
                Forecast request
                      ↓
              "Next 7 days"
                      ↓
            Get current date
                      ↓
        Create calendar features
                      ↓
      ┌───────────────────────────┐
      │ Day 1                     │
      │                            │
      │ Day of week               │
      │ Month                     │
      │ Weekend?                  │
      │ Promotion?                │
      │ Day of year               │
      │ Yesterday's demand        │
      │ 7-day lag                 │
      │ 7-day average             │
      └─────────────┬─────────────┘
                    ↓
            Gradient Boosting
                    ↓
              Forecast Day 1
                    ↓
            Update lag values
                    ↓
      ┌───────────────────────────┐
      │ Day 2                     │
      └─────────────┬─────────────┘
                    ↓
               Predict
                    ↓
             Update history
                    ↓
                  ...
                    ↓
              Forecast Day 7
                    ↓
          Add all daily forecasts
                    ↓
             Total demand
```

---

# 47. What is the model actually learning?

The model is learning relationships such as:

```text
Weekend?
    ↓
Demand tends to be higher

Promotion?
    ↓
Demand tends to be higher

Recent demand high?
    ↓
Tomorrow may also be higher

Last week's demand high?
    ↓
Today may be higher

Recent average high?
    ↓
Demand may remain high
```

So instead of manually writing:

```text
if weekend:
    add 25

if promotion:
    add 40
```

the model learns patterns from the training data.

---

# 48. Why use machine learning instead of simply using the formula?

Because in a real business you usually don't know the exact formula.

You might have hundreds of factors:

```text
Day
Weather
Promotion
Price
Holiday
Competitor price
Store
Region
Inventory
Marketing
Previous sales
Season
...
```

A machine-learning model can learn relationships from historical data instead of requiring you to manually write every rule.

Your prototype demonstrates that idea with a smaller set of features.

---

# 49. Why Gradient Boosting instead of ARIMA?

This is an interview question you might get.

A simple answer:

> **"I chose Gradient Boosting because I wanted to combine historical sales features with external variables such as promotions and calendar information. Tree-based models can also capture nonlinear interactions between these features."**

For example:

```text
Weekend + Promotion
```

could have a very different effect from:

```text
Weekday + No Promotion
```

Gradient Boosting can learn these interactions.

---

# 50. But don't say ARIMA "cannot use external variables"

Your original documentation says:

> "ARIMA cannot easily incorporate external promotional flags."

I'd phrase that more carefully.

ARIMA itself is mainly designed around the time-series structure, but models such as **ARIMAX/SARIMAX** can incorporate exogenous variables.

So in an interview, say:

> **"I chose a tree-based supervised approach because it lets me directly combine lag features, calendar features, and promotion flags in one model."**

That's more technically accurate.

---

# 51. What is actually implemented?

Based on your code:

### ✅ Implemented

* 500 days of synthetic sales data
* Daily demand
* Weekly/weekend effects
* Yearly seasonal effect
* Promotion effect
* Gradual growth trend
* Random demand noise
* `lag_1`
* `lag_7`
* 7-day rolling mean
* Chronological 80/20 train/test split
* Gradient Boosting Regressor
* 100 estimators
* Maximum tree depth of 4
* MAE evaluation
* R² evaluation
* Multi-day recursive forecasting
* Total forecast calculation
* Simple ±12% planning range
* Model serialization

### ❌ Not actually implemented

* Multiple stores
* Store-specific models
* Real POS data
* Real inventory data
* Real promotions
* Real holidays
* Weather
* Pricing
* Competitor information
* Automatic purchase orders
* ERP/WMS integration
* Weekly automated retraining
* True statistical prediction intervals
* Proper dynamically updated `lag_7`

Those can be production improvements, but they aren't in this code.

---

# 52. One particularly important issue: `store_id`

Your API accepts:

```json
{
  "store_id": "STORE_101"
}
```

But the model doesn't use it.

So these two requests:

```text
STORE_101
```

and:

```text
STORE_999
```

will use exactly the same model and features.

The `store_id` is basically just returned in the response.

If you actually wanted different stores to have different demand patterns, you'd need to incorporate store information into the training data/model.

---

# 53. Another important issue: the forecast starts from `datetime.now()`

Training data starts at:

```text
January 1, 2025
```

But prediction uses:

```python
start_date = datetime.now()
```

So the model is trained on data from 2025 but may be asked to predict from the current date.

That's okay for a demonstration, but it creates a mismatch.

A real system would normally train on recent historical data and forecast from the actual latest known date.

---

# 54. Another important issue: the promotion assumption

Your API accepts:

```json
"is_promo": 1
```

and then uses that same value for **every forecast day**.

So:

```text
Day 1 → promotion
Day 2 → promotion
Day 3 → promotion
Day 4 → promotion
...
```

if you send:

```text
is_promo = 1
```

That's not necessarily how real promotions work.

A production API might instead accept something like:

```text
Monday    → no promotion
Tuesday   → no promotion
Wednesday → promotion
Thursday  → promotion
Friday    → no promotion
```

But your current version uses one promotion flag for the whole forecast horizon.

---

# 55. The easiest way to understand the lag features

Remember this:

```text
lag_1
↓
What happened yesterday?

lag_7
↓
What happened one week ago?

rolling_mean_7
↓
What has demand looked like recently?
```

That's basically the heart of the time-series part.

---

# 56. The easiest way to understand the whole project

Imagine you are a store manager.

You ask your ML system:

> **"How many products should I expect to sell during the next 7 days?"**

The system looks at:

```text
📅 What day is it?
📅 What month is it?
🌤️ Is it a weekend?
🏷️ Is there a promotion?
📈 How much did we recently sell?
📅 What did we sell around this time last week?
📊 What's our recent average?
```

Then:

```text
             ↓

       ML Model

             ↓

Monday    → 120
Tuesday   → 135
Wednesday → 142
Thursday  → 110
Friday    → 95
Saturday  → 205
Sunday    → 210
```

Then:

```text
Total expected demand
        ↓
      1,017 units
```

Now the business knows roughly how much stock it may need.

---

# 57. How I would explain it in an interview

If someone asks:

### "Tell me about your demand forecasting project."

I'd say:

> **"I built a demand forecasting pipeline using a Gradient Boosting Regressor. The goal is to predict daily product demand for future dates so that a business can make better inventory decisions.**
>
> **For the prototype, I generated 500 days of synthetic sales data with seasonality, weekend effects, promotions, a gradual trend, and random noise. I then created time-series features such as yesterday's sales, sales from seven days ago, and the seven-day rolling average.**
>
> **I used a chronological 80/20 train-test split rather than shuffling the data because future information shouldn't be used to predict the past. I trained a Gradient Boosting Regressor and evaluated it using MAE and R².**
>
> **For inference, the model recursively predicts each future day. The prediction for one day is fed back as an input for subsequent days. The API returns daily forecasts, total expected demand, and a simple ±12% planning range."**

---

# 58. If they ask "Why is it called time-series forecasting?"

Say:

> **"Because the observations are ordered in time, and previous demand values are used as features to predict future demand."**

That's the key definition.

---

# 59. If they ask "What is lag?"

Say:

> **"A lag is a previous value of the target variable. In my model, lag_1 is yesterday's sales and lag_7 is sales from seven days earlier."**

Simple.

---

# 60. If they ask "What is recursive forecasting?"

Say:

> **"Instead of predicting all future days independently, I predict one day first, feed that prediction into the next day's lag features, and continue iteratively across the forecast horizon."**

That's exactly what your code is trying to do.

---

# 61. If they ask "What is MAE?"

Say:

> **"MAE is Mean Absolute Error. It tells me, in the same unit as my target, how far the predictions are from actual demand on average. So an MAE of 14.5 means the model is off by roughly 14.5 units per day on average."**

---

# 62. The one-line version

If you remember only one thing, remember:

> **Past sales + calendar + promotions → Gradient Boosting → Future daily demand**

And your six projects now look like this:

```text
🛡️ Fraud Detection
Transaction
    ↓
Random Forest
    ↓
Fraud probability
    ↓
Risk decision


💳 Credit Risk
Borrower
    ↓
Logistic Regression
    ↓
Default probability
    ↓
Loan decision


📉 Customer Churn
Customer
    ↓
Gradient Boosting
    ↓
Churn probability
    ↓
Retention action


🏡 House Price
House
    ↓
Gradient Boosting Regressor
    ↓
Predicted price


🎯 Recommendation
User ratings
    ↓
SVD
    ↓
Predicted preferences
    ↓
Recommended products


📊 Demand Forecasting
Past sales + time + promotion
    ↓
Gradient Boosting Regressor
    ↓
Future demand
    ↓
Inventory planning
```

The key difference to remember is:

**House Price predicts a value for a property. Demand Forecasting predicts future values along a timeline, so previous sales become part of the input.**
