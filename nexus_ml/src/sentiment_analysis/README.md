# 💬 Sentiment Analysis

## Beginner-Friendly Explanation

## 1. What is this project?

Imagine a customer writes:

> "The product is amazing and delivery was super fast!"

Your system needs to answer:

```text
POSITIVE
```

If they write:

> "The product broke after one day. Terrible quality."

The system should answer:

```text
NEGATIVE
```

And if they write:

> "The product arrived today. It works as expected."

It might answer:

```text
NEUTRAL
```

So the entire project is basically:

```text
Customer text
     ↓
Understand the words
     ↓
Convert text into numbers
     ↓
ML model
     ↓
POSITIVE / NEUTRAL / NEGATIVE
```

This is called **sentiment classification**.

---

# 2. Why can't we just give text directly to Logistic Regression?

This is the first important concept.

A machine-learning model doesn't naturally understand:

```text
"The product is amazing"
```

as humans do.

It needs numbers.

So we need to convert:

```text
"The product is amazing"
```

into something like:

```text
[0.0, 0.0, 0.73, 0.12, 0.0, ...]
```

That's what **TF-IDF** does.

So your system actually has two major components:

```text
TEXT
 ↓
TF-IDF
 ↓
NUMBERS
 ↓
LOGISTIC REGRESSION
 ↓
SENTIMENT
```

---

# 3. What data are you training on?

Your code creates synthetic review data.

You have three groups of phrases.

### Positive

```text
"absolutely amazing product"
"best quality ever"
"fast shipping works great"
"highly recommend to everyone"
"fantastic customer support"
"exceeded my expectations"
```

And each one gets:

```text
POSITIVE
```

---

### Neutral

```text
"received the order today"
"item arrived as expected"
"average build quality okay"
"nothing special but works fine"
"standard shipping time"
```

Each gets:

```text
NEUTRAL
```

---

### Negative

```text
"terrible terrible quality broke"
"worst purchase waste of money"
"horrible customer service"
"do not buy defective item"
"extremely disappointed"
```

Each gets:

```text
NEGATIVE
```

---

# 4. Why do you combine two phrases?

Your code does:

```python
text = random_positive_phrase + " " + random_positive_phrase
```

So instead of having:

```text
"absolutely amazing product"
```

you might get:

```text
"absolutely amazing product fast shipping works great"
```

This creates a slightly larger training example.

You do the same for neutral and negative text.

---

# 5. So what does the training dataset look like?

Eventually you get something like:

```text
TEXT                                      LABEL

"absolutely amazing product ..."          POSITIVE

"fast shipping works great ..."           POSITIVE

"received the order today ..."            NEUTRAL

"standard shipping time ..."              NEUTRAL

"terrible terrible quality ..."           NEGATIVE

"worst purchase waste of money ..."       NEGATIVE
```

The model's job is:

> **Learn which patterns in the text are associated with each label.**

---

# 6. Now comes TF-IDF

This is probably the most confusing part.

Don't think of TF-IDF as complicated mathematics.

Think of it as:

> **A way of deciding which words/phrases are important in a piece of text.**

For example, imagine we have:

```text
"The product is amazing"
```

Words like:

```text
product
```

might appear in lots of reviews.

But:

```text
amazing
```

may be much more useful for identifying positive sentiment.

TF-IDF tries to give more importance to useful terms and less importance to terms that appear everywhere.

---

# 7. What does TF mean?

TF means:

> **Term Frequency**

Basically:

> How often does this word appear in this particular document?

Suppose the review is:

```text
"amazing product, amazing quality"
```

The word:

```text
amazing
```

appears twice.

So it gets a stronger term-frequency signal.

---

# 8. What does IDF mean?

IDF means:

> **Inverse Document Frequency**

The idea is:

> If a word appears in almost every document, it's not very useful for distinguishing documents.

For example:

```text
"product"
```

might appear in:

```text
positive reviews
neutral reviews
negative reviews
```

So it's not very helpful for deciding sentiment.

But:

```text
"terrible"
```

might mostly appear in negative reviews.

That's much more useful.

So TF-IDF gives more weight to distinctive words.

---

# 9. An easy example

Imagine we have these three reviews:

```text
1. "amazing product"
2. "average product"
3. "terrible product"
```

The word:

```text
product
```

appears everywhere.

So:

```text
product → less useful
```

But:

```text
amazing → positive signal
average → neutral signal
terrible → negative signal
```

So TF-IDF helps the model discover these patterns.

---

# 10. But your code does something even better

You use:

```python
ngram_range=(1, 2)
```

This means:

> Look at individual words **and two-word combinations**.

These are called:

### Unigrams

One word:

```text
amazing
great
terrible
quality
```

### Bigrams

Two consecutive words:

```text
very good
terrible quality
works great
waste money
```

So your model doesn't just look at individual words.

It also looks at short phrases.

---

# 11. Why are bigrams useful?

Imagine:

```text
"good"
```

and:

```text
"not good"
```

Those two sentences contain the word:

```text
good
```

But they have completely different meanings.

Looking only at individual words can cause problems.

With bigrams, the model can learn:

```text
"not good"
```

as a separate feature.

That's why:

```python
ngram_range=(1, 2)
```

is useful.

---

# 12. What does `max_features=2500` mean?

You have:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=2500
)
```

Imagine your dataset contains thousands of different words and phrases.

You don't necessarily want every possible one.

So you tell TF-IDF:

> **Keep at most 2,500 features.**

Conceptually:

```text
All words + phrases
        ↓
    TF-IDF
        ↓
Keep important vocabulary
        ↓
Maximum 2,500 features
```

This keeps the model smaller and faster.

---

# 13. Now the text becomes numbers

Suppose the vocabulary contains:

```text
amazing
great
terrible
quality
works great
terrible quality
```

The sentence:

```text
"terrible quality"
```

gets converted into numerical features.

Conceptually:

```text
"terrible quality"
        ↓
[0, 0, 1, ..., 0, 1]
```

The actual values aren't simply 0 and 1; they're TF-IDF weights.

But the important idea is:

> **Text → numerical vector**

---

# 14. Then Logistic Regression takes over

Now we have:

```text
Text
 ↓
TF-IDF
 ↓
Numerical vector
 ↓
Logistic Regression
```

The Logistic Regression model learns which words and phrases are associated with each sentiment.

For example, conceptually:

```text
"amazing"          → POSITIVE
"fantastic"        → POSITIVE
"works great"      → POSITIVE

"average"          → NEUTRAL
"as expected"      → NEUTRAL

"terrible"         → NEGATIVE
"waste of money"   → NEGATIVE
"defective"        → NEGATIVE
```

The model actually learns numerical weights rather than using a simple dictionary, but this is the easiest way to understand what's happening.

---

# 15. Why is it called Logistic Regression if there are 3 classes?

Your problem has:

```text
POSITIVE
NEUTRAL
NEGATIVE
```

That's **multi-class classification**.

Scikit-learn's Logistic Regression can handle multiple classes.

So it learns to distinguish among:

```text
POSITIVE
NEUTRAL
NEGATIVE
```

rather than just:

```text
YES
NO
```

---

# 16. The entire training pipeline

Your code does this:

```text
1,200 reviews
      ↓
Positive / Neutral / Negative labels
      ↓
Split into training and testing data
      ↓
TF-IDF
      ↓
Convert text → numbers
      ↓
Logistic Regression
      ↓
Learn sentiment patterns
      ↓
Test the model
      ↓
Save the complete pipeline
```

---

# 17. Why do you use `SkPipeline`?

You have:

```python
pipeline = SkPipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf", LogisticRegression(...))
])
```

This is actually a nice engineering choice.

Instead of separately saving:

```text
TF-IDF
+
Logistic Regression
```

you combine them into one object.

So later you can simply give it:

```text
"The product is amazing"
```

and the pipeline automatically does:

```text
Text
 ↓
TF-IDF
 ↓
Numbers
 ↓
Logistic Regression
 ↓
Prediction
```

You don't have to manually call the vectorizer first.

---

# 18. This is why saving the pipeline is useful

Your code does:

```python
self.model = {
    "pipeline": pipeline,
    "metrics": ...
}
```

So the saved model contains the entire process.

When a new review comes in:

```text
New text
  ↓
Saved TF-IDF vocabulary
  ↓
Saved Logistic Regression
  ↓
Prediction
```

The preprocessing stays consistent with training.

---

# 19. What happens when a new review arrives?

Suppose the API receives:

```text
"The product exceeded my expectations! Super fast delivery and great quality."
```

Your code does:

```python
text = input_data.get("text", "")
```

Then:

```python
probs = pipeline.predict_proba([text])[0]
```

This asks:

> **What probability does the model assign to each sentiment?**

For example, conceptually:

```text
POSITIVE → 0.96
NEUTRAL  → 0.03
NEGATIVE → 0.01
```

---

# 20. How does it choose the final sentiment?

You do:

```python
best_idx = np.argmax(probs)
```

`argmax` means:

> **Find the biggest number.**

Here:

```text
POSITIVE → 0.96  ← biggest
NEUTRAL  → 0.03
NEGATIVE → 0.01
```

So:

```text
sentiment = POSITIVE
```

---

# 21. What is confidence?

You then do:

```python
confidence = float(probs[best_idx])
```

Since the winning probability was:

```text
0.96
```

you return:

```text
confidence = 0.96
```

So your API might say:

```json
{
  "sentiment": "POSITIVE",
  "confidence": 0.96
}
```

In simple language:

> **The model's highest class probability is 96%.**

Be slightly careful about calling this "true confidence"; it's better understood as the model's predicted probability for its selected class unless you've separately validated calibration.

---

# 22. What is the composite sentiment score?

This is one of the cleverer parts of your code.

You calculate:

```python
composite_score = (
    probs[0] - probs[2]
)
```

Suppose:

```text
POSITIVE = 0.90
NEGATIVE = 0.05
```

Then:

```text
0.90 - 0.05 = +0.85
```

So:

```text
+0.85
```

means strongly positive.

---

# 23. What if the review is negative?

Suppose:

```text
POSITIVE = 0.05
NEUTRAL  = 0.10
NEGATIVE = 0.85
```

Then:

```text
0.05 - 0.85 = -0.80
```

So:

```text
-0.80
```

means strongly negative.

---

# 24. What does 0 mean?

Suppose:

```text
POSITIVE = 0.40
NEGATIVE = 0.40
```

Then:

```text
0.40 - 0.40 = 0
```

So the score is neutral between positive and negative sentiment.

The scale is approximately:

```text
-1.0                       0                       +1.0
 │                         │                         │
Very negative            Balanced               Very positive
```

---

# 25. Notice that NEUTRAL isn't directly included

Your formula is:

```python
P(POSITIVE) - P(NEGATIVE)
```

It doesn't subtract or add the neutral probability.

For example:

```text
POSITIVE = 0.10
NEUTRAL  = 0.80
NEGATIVE = 0.10
```

gives:

```text
0.10 - 0.10 = 0
```

That's reasonable because positive and negative evidence are balanced.

But the `sentiment` field can still be:

```text
NEUTRAL
```

because NEUTRAL had the highest probability.

So there are actually two different outputs:

```text
sentiment
    ↓
Most likely class

composite_score
    ↓
Positive probability minus negative probability
```

---

# 26. What happens if someone sends empty text?

Your code checks:

```python
if not text.strip():
```

This means:

> Is the text empty or only spaces?

If yes, you return:

```text
NEUTRAL
confidence = 0.50
score = 0.0
```

That's basically a safe fallback.

---

# 27. What does `train_test_split` do?

You don't train on all 1,200 reviews.

You split them:

```text
80%
 ↓
Training data

20%
 ↓
Testing data
```

The model learns from the first group.

Then you test whether it can classify reviews it didn't train on.

---

# 28. Why use `stratify=y`?

You have three classes:

```text
POSITIVE
NEUTRAL
NEGATIVE
```

You don't want your random split to accidentally produce something like:

```text
Training:
400 positive
400 neutral
160 negative

Testing:
0 positive
0 neutral
240 negative
```

Instead, `stratify=y` tries to preserve the class proportions in both sets.

So the split remains balanced.

---

# 29. What is accuracy?

You calculate:

```python
accuracy_score(y_test, y_pred)
```

Accuracy means:

> **What percentage of test reviews did the model classify correctly?**

If there are 240 test reviews and 240 are correct:

```text
240 / 240 = 100%
```

So:

```text
Accuracy = 1.00
```

---

# 30. What is F1-score?

You also calculate:

```python
f1_score(
    y_test,
    y_pred,
    average="weighted"
)
```

F1 combines:

```text
Precision
+
Recall
```

into one score.

The `weighted` option means classes with more examples have more influence on the final score.

In your case, because your generated dataset is approximately balanced between the three classes, this won't be dramatically different from a simple average.

---

# 31. Why does this model get 100% accuracy?

This is a **very important limitation** of your project.

You created the training data yourself from very obvious phrases.

For example:

```text
"absolutely amazing"
"fantastic"
"highly recommend"
```

are obviously positive.

And:

```text
"terrible"
"worst purchase"
"defective"
```

are obviously negative.

So the model has an extremely easy job.

It's basically learning:

```text
amazing → positive
terrible → negative
average → neutral
```

Therefore:

```text
Accuracy ≈ 100%
```

is not evidence that this model would get 100% accuracy on real customer reviews.

---

# 32. This is an important interview answer

If someone asks:

> **"Your sentiment model has 100% accuracy. Is it really that good?"**

Don't say:

> "Yes, it is extremely accurate."

Instead say:

> **"The model achieves approximately 100% on our synthetic test set, but that result is not representative of real-world performance. The synthetic phrases are highly separable and were generated from the same predefined sentiment categories. A real deployment would require a large, human-labeled dataset containing ambiguity, slang, misspellings, sarcasm, negation and domain-specific language."**

That's a much stronger answer.

---

# 33. Why TF-IDF instead of a Transformer?

For this project, TF-IDF + Logistic Regression is a reasonable lightweight baseline.

Imagine you need to process:

```text
10 million reviews
```

You could use a large transformer model, but that can require substantially more compute.

Your approach is:

```text
Review
 ↓
TF-IDF
 ↓
Small numerical vector
 ↓
Logistic Regression
```

It's relatively:

```text
Fast
Cheap
Simple
Easy to deploy
Easy to inspect
```

The trade-off is that it doesn't understand language as deeply as modern transformer models.

---

# 34. What can TF-IDF struggle with?

This is another good interview topic.

### Sarcasm

```text
"Wow, fantastic. My product broke on day one."
```

A simple model may struggle.

### Context

```text
"This isn't bad."
```

The word:

```text
bad
```

looks negative, but the overall meaning can be positive.

### Long-range relationships

```text
"I expected the product to be terrible, but after using it for a month I absolutely love it."
```

TF-IDF isn't really understanding the whole sentence the way a modern language model might.

### Synonyms

People can express the same feeling in many different ways.

---

# 35. Why do bigrams help?

Your:

```python
ngram_range=(1, 2)
```

allows the model to learn phrases such as:

```text
"works great"
"terrible quality"
"waste money"
"very satisfied"
```

instead of only individual words.

That's especially useful for sentiment.

But it's still not the same thing as truly understanding language.

---

# 36. Your complete architecture

Here's the whole project:

```text
                    CUSTOMER REVIEW
                           │
                           ↓
                    "Amazing product!"
                           │
                           ↓
                    ┌─────────────┐
                    │   TF-IDF    │
                    └──────┬──────┘
                           │
                     Numbers
                           │
                           ↓
                ┌───────────────────┐
                │ Logistic Regression│
                └─────────┬─────────┘
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
           POSITIVE    NEUTRAL     NEGATIVE
             0.96        0.03         0.01
              │
              ↓
         POSITIVE
              │
              ↓
       Confidence = 0.96
              │
              ↓
Composite Score = 0.96 - 0.01
                = 0.95
```

---

# 37. The most important concept to remember

Your project has **three stages**:

### Stage 1 — Convert language to numbers

```text
TF-IDF
```

### Stage 2 — Classify those numbers

```text
Logistic Regression
```

### Stage 3 — Turn the model output into useful information

```text
Sentiment
+
Confidence
+
Composite score
```

That's the whole project.

---

# 38. How I'd explain it to someone who knows nothing about ML

I'd say:

> **"We have a system that reads customer reviews and decides whether the customer is positive, neutral or negative. Computers can't directly give normal sentences to a traditional machine-learning model, so first we convert the important words and short phrases into numbers using TF-IDF. Then Logistic Regression learns which numerical patterns are associated with positive, neutral and negative reviews. When a new review arrives, the same TF-IDF transformation is applied, the model calculates probabilities for all three sentiments, and we choose the sentiment with the highest probability. We also calculate a score by subtracting the negative probability from the positive probability, giving us a rough measure from negative to positive."**

---

# 39. How I'd explain it in an interview

If they say:

### "Walk me through your sentiment analysis project."

Say:

> **"I built a three-class sentiment classification pipeline for positive, neutral and negative customer reviews. Because the input is raw text, I first use TF-IDF to convert the text into numerical features. I use both unigrams and bigrams so the model can capture individual words as well as short phrases. I then feed those features into Logistic Regression, which learns the relationship between text patterns and sentiment classes.**
>
> **I split the dataset into stratified training and test sets and evaluate using accuracy and weighted F1. During inference, the saved pipeline transforms the incoming text using the same TF-IDF vocabulary and predicts probabilities for all three classes. I return the highest-probability sentiment, its probability, and a composite sentiment score calculated as positive probability minus negative probability."**

---

# 40. If they ask "Why TF-IDF?"

Say:

> **"TF-IDF is a lightweight way to represent text numerically while giving more importance to words and phrases that are informative for distinguishing documents. It's fast, simple and works well as a baseline for text classification."**

---

# 41. If they ask "Why bigrams?"

Say:

> **"Unigrams capture individual words, while bigrams capture two-word phrases. Bigrams help preserve some local context, for example distinguishing phrases like 'good' and 'not good' or recognizing phrases like 'terrible quality'."**

---

# 42. If they ask "Why Logistic Regression?"

Say:

> **"It's a strong lightweight baseline for sparse text features. TF-IDF produces a high-dimensional sparse feature matrix, and Logistic Regression handles that representation efficiently while providing class probabilities."**

---

# 43. If they ask "What does `predict_proba()` do?"

Say:

> **"It gives the model's estimated probability for each sentiment class. For example, it might return 0.90 positive, 0.08 neutral and 0.02 negative. We choose the class with the highest probability."**

---

# 44. If they ask "What is the biggest weakness?"

Say:

> **"The biggest weakness is the synthetic dataset. The vocabulary is very clean and strongly correlated with the labels, so the 100% test accuracy is optimistic. Real customer language contains sarcasm, ambiguity, spelling errors, mixed sentiment and domain-specific expressions. I'd validate the approach on a human-labeled real-world dataset before considering production deployment."**

---

# 45. The one diagram you should memorize

```text
             REVIEW
               ↓
            TF-IDF
               ↓
       WORDS → NUMBERS
               ↓
      LOGISTIC REGRESSION
               ↓
      ┌────────┼────────┐
      ↓        ↓        ↓
   POSITIVE  NEUTRAL  NEGATIVE
    0.90      0.08      0.02
      │
      ↓
  POSITIVE
      │
      ↓
Confidence = 90%

Composite score:
0.90 - 0.02 = +0.88
```

If you understand **that diagram**, you understand the core of your entire sentiment-analysis code.
