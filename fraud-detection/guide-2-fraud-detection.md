# Guide 2: Credit Card Fraud Detection (Classification)

**Goal:** Given a transaction's details, predict whether it's fraud or not.
This is a classification problem — the answer is a category (Fraud / Not
Fraud), not a number.

Same structure as Guide 1: follow top to bottom, run every cell yourself,
read every explanation before moving on.

---

## 0. The big picture (read this first)

This dataset has a twist that makes it a genuinely important lesson: out of
284,807 transactions, only 492 are fraud — that's about **0.17%**. If a
model just predicted "not fraud" for everything, it would be 99.8% accurate
and completely useless. This is called **class imbalance**, and it means
**accuracy is the wrong metric here**. We'll use precision, recall, F1, and
ROC-AUC instead — and explain exactly why each one matters more than
accuracy for this problem.

---

## 1. Folder structure

```
fraud-detection/
├── data/
│   └── raw/                  # original CSV, never edited by hand
├── notebooks/
│   └── 01_model.ipynb
├── models/
│   └── model.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 2. GitHub setup

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
mkdir -p data/raw notebooks models app
```

(Create the repo on github.com first, same as Guide 1: **New repository**
→ name it `fraud-detection` → add a README → create.)

---

## 3. Local environment setup

```bash
python -m venv venv
```

Activate:
- Mac/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\Activate.ps1`

`requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
joblib
streamlit
```

```bash
pip install -r requirements.txt
```

Same package purposes as Guide 1 — nothing new here except how we'll *use*
scikit-learn's metrics module, which we cover in Part J.

```bash
git add .
git commit -m "Initial project structure and requirements"
git push
```

---

## 4. Get the data

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download it (free Kaggle account required)
3. Unzip and place `creditcard.csv` in `data/raw/`

Add a `.gitignore` in the project root:

```
venv/
data/raw/*.csv
```

We keep raw data out of git for the same reason as before: it's not our
data to redistribute, and it's large.

---

## 5. Launch Jupyter

```bash
jupyter notebook
```

Create `notebooks/01_model.ipynb`. Every code block below is one cell, run
in order.

---

## Part A — Load and inspect the data

```python
import pandas as pd

df = pd.read_csv("../data/raw/creditcard.csv")

print(df.shape)
df.head()
```

**What's in this file:**
- `Time` — seconds elapsed between this transaction and the first one in
  the dataset
- `Amount` — transaction amount
- `V1` through `V28` — the actual transaction details (merchant, location,
  etc.), but scrambled through a privacy technique called PCA so the bank's
  real customer data can never be reconstructed from it. We can't read what
  they mean individually, but the model can still learn patterns from them.
- `Class` — the answer: `1` = fraud, `0` = legitimate

```python
df.info()
```

**Why this matters here specifically:** confirms every column is numeric
(no messy text to clean) and shows if any values are missing.

---

## Part B — Data cleaning (skip nothing)

### B1. Duplicates

```python
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)
```

**Why this matters extra here:** with only 492 fraud cases total, even a
handful of duplicated fraud rows would meaningfully inflate how much
"fraud pattern" the model thinks it's seeing, and would let the same exact
row leak into both the train and test sets — making the model look better
than it really is.

### B2. Missing values

```python
print(df.isnull().sum().sum())  # total missing values across the whole table
```

If this prints `0`, there's genuinely nothing to fix — that's a valid,
expected outcome, not a step to skip explaining. This dataset is
pre-cleaned by its publisher, so an empty result here is normal; you
still run the check because you never assume a dataset is clean without
looking.

### B3. Correct data types

```python
print(df.dtypes)
```

Everything should already be numeric (`int64` / `float64`). If any column
shows as `object` (text), that would signal a formatting problem worth
investigating before moving on.

### B4. Outlier check on Amount

```python
df["Amount"].describe()
```

```python
import matplotlib.pyplot as plt

plt.boxplot(df["Amount"])
plt.title("Transaction Amount spread")
plt.show()
```

**Reading this:** transaction amounts are naturally skewed — most
purchases are small, a few are large. That's expected here, not an error
to fix by deleting rows. We'll handle this skew properly in Part F, not by
throwing away real transactions.

### B5. Class balance check — the most important cleaning step in this project

```python
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True) * 100)
```

**Why this is a "cleaning" step and not just an EDA curiosity:** this
number changes every decision from here on — which metric we trust, how
we split the data, and how we train the model. Confirm it now, in writing,
before doing anything else.

---

## Part C — Exploratory Data Analysis (EDA)

```python
import seaborn as sns

sns.countplot(x="Class", data=df)
plt.title("Fraud (1) vs Legitimate (0) transaction counts")
plt.show()
```

**What you'll see:** a bar for `0` towering over a nearly invisible bar for
`1`. This chart is the clearest possible picture of why accuracy alone
would lie to us — visually, "predict everything as 0" looks almost
correct.

```python
df.groupby("Class")["Amount"].describe()
```

**What to look for:** compare the average (`mean`) transaction amount for
fraud vs legitimate. If they differ noticeably, `Amount` alone carries some
signal the model can use.

---

## Part D — Feature engineering

```python
# Time and Amount are on very different scales than V1-V28
# (which are already roughly standardized by the PCA process).
# We scale them so no single feature dominates just because of its raw size.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_time"] = scaler.fit_transform(df[["Time"]])

df = df.drop(["Amount", "Time"], axis=1)
```

**Why scale at all?** Many models (and especially anything using
distances or gradients) can be thrown off if one feature ranges from
0–25,000 (`Amount`) while others range from -5 to 5 (`V1`–`V28`). Scaling
puts everything on a comparable footing so the model weighs features by
their actual predictive value, not their raw size.

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

---

## Part E — Train/test split (done carefully, because of the imbalance)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why `stratify=y` matters here specifically:** with only 492 fraud cases
in the whole dataset, a plain random split could easily land most of them
in the training set and leave the test set with almost none — making
evaluation meaningless. `stratify=y` forces both the train and test sets
to keep the same fraud/legitimate ratio as the original data.

```python
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

Confirm both splits show roughly the same ~0.17% fraud rate.

---

## Part F — Baseline model: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)
```

**Why `class_weight="balanced"`?** Without it, a model minimizing overall
error will happily ignore the rare fraud class since getting those 0.17%
wrong barely dents its score. This setting tells the model to treat
mistakes on the rare class as more costly, roughly in proportion to how
rare it is — forcing it to actually pay attention to fraud cases instead
of coasting on the majority class.

---

## Part G — Better model: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
```

**Why try this too?** Random Forests can capture more complex,
non-straight-line patterns across the 28 anonymized features than Logistic
Regression can. We keep `class_weight="balanced"` for the same reason as
above.

---

## Part H — Why accuracy is misleading here (prove it yourself)

```python
from sklearn.metrics import accuracy_score

print("Baseline accuracy:", accuracy_score(y_test, baseline_predictions))
print("Random Forest accuracy:", accuracy_score(y_test, rf_predictions))
```

Both will likely print something like `0.97`–`0.99`. **That looks great
and tells you almost nothing.** A model that predicted "not fraud" for
every single row would also score around 99.8% accuracy on this data,
while catching zero fraud. This is exactly why we need the metrics in
Part J.

---

## Part I — Confusion matrix (see the actual mistakes)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, rf_predictions)
ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot()
plt.title("Random Forest — Confusion Matrix")
plt.show()
```

**Reading this grid — the four outcomes possible on any single
prediction:**
- **True Negative** — actually legit, correctly predicted legit
- **False Positive** — actually legit, wrongly flagged as fraud (annoys a
  real customer, blocks a real payment)
- **False Negative** — actually fraud, wrongly predicted legit (the bank
  loses money — this is usually the worst mistake in fraud detection)
- **True Positive** — actually fraud, correctly caught

Everything in Part J is really just different ways of summarizing this
grid into a single number.

---

## Part J — The metrics that actually matter here

```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

def evaluate(name, y_true, y_pred, y_proba):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"{name}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    print(f"  ROC-AUC:   {roc_auc:.3f}")

baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]

evaluate("Logistic Regression", y_test, baseline_predictions, baseline_proba)
evaluate("Random Forest", y_test, rf_predictions, rf_proba)
```

**What each metric means, in plain words:**

- **Precision** — of everything the model flagged as fraud, what
  fraction actually was fraud? Low precision means lots of false alarms
  on real customers.
  `Precision = True Positives / (True Positives + False Positives)`

- **Recall** — of all the real fraud cases that happened, what fraction
  did the model actually catch? Low recall means fraud is slipping
  through undetected.
  `Recall = True Positives / (True Positives + False Negatives)`

- **F1-score** — a single number that balances precision and recall
  together. Useful when you want one score to compare models, since
  chasing precision alone or recall alone can each be misleading on their
  own.

- **ROC-AUC** — measures how well the model separates fraud from
  legitimate transactions *across every possible decision threshold*, not
  just the default 50% cutoff. Ranges from 0.5 (no better than guessing)
  to 1.0 (perfect separation). This is often the headline metric for
  fraud detection because it doesn't depend on picking one specific
  cutoff point.

**Why precision and recall trade off against each other:** if you make
the model more cautious about flagging fraud (raising the bar), you catch
fewer false alarms (precision goes up) but also miss more real fraud
(recall goes down) — and vice versa. There's no free lunch here; the
right balance depends on whether the bank cares more about blocking real
customers or letting fraud through. This is a business decision, not
just a modeling one.

```python
print(classification_report(y_test, rf_predictions, target_names=["Legit", "Fraud"]))
```

This prints precision, recall, and F1 for both classes at once — a
quick full summary.

---

## Part K — Save the model

```python
import joblib

# Save the better-performing model — assume Random Forest here
joblib.dump(rf_model, "../models/model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")  # needed to scale new inputs the same way
```

**Why save the scaler too?** Any new transaction the app receives later
needs to go through the *exact same* scaling transformation the training
data went through — otherwise the model is comparing numbers on
completely different scales than it learned from.

```bash
git add .
git commit -m "Add cleaning, imbalance-aware modeling and evaluation"
git push
```

---

## Part L — Streamlit app

Create `app/app.py`. Since `V1`–`V28` are anonymized and meaningless to
type in by hand, this app lets the user adjust `Amount` and `Time`, and
uses average values for `V1`–`V28` so the demo stays usable and honest
about what a real user could realistically input.

```python
import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.title("Credit Card Fraud Detector")
st.write(
    "Enter a transaction's amount and time to estimate fraud risk. "
    "(V1-V28 are anonymized bank-internal features set to typical "
    "values for this demo, since a real user can't provide them directly.)"
)

amount = st.number_input("Transaction amount", min_value=0.0, value=50.0)
time = st.number_input("Seconds since first transaction in dataset", min_value=0, value=10000)

if st.button("Check transaction"):
    # Scale amount and time the same way training data was scaled
    scaled_amount = scaler.transform([[amount]])[0][0]
    scaled_time = scaler.transform([[time]])[0][0]

    # V1-V28 set to 0 as a neutral placeholder (their scaled average)
    v_features = [0.0] * 28

    input_row = pd.DataFrame(
        [v_features + [scaled_amount, scaled_time]],
        columns=[f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"],
    )

    prediction = model.predict(input_row)[0]
    probability = model.predict_proba(input_row)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely fraud — risk score: {probability:.2%}")
    else:
        st.success(f"✅ Likely legitimate — risk score: {probability:.2%}")
```

**Why call out the V1–V28 simplification instead of hiding it?** A real
fraud system would compute those 28 features automatically from
merchant, location, and behavior data behind the scenes — a demo app
can't recreate that pipeline, and pretending otherwise would be
misleading. Being upfront about a demo's limits is itself good practice.

Run it locally:

```bash
cd app
streamlit run app.py
```

---

## Part M — Deploy to Streamlit Community Cloud

1. Commit and push `app/app.py`, `models/model.pkl`, `models/scaler.pkl`,
   and `requirements.txt` to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. **New app** → pick the `fraud-detection` repo → branch `main` → file
   path `app/app.py`.
4. **Deploy**.

Same path-handling snag as Guide 1 can happen here: if Streamlit Cloud
can't find `model.pkl`, switch the load paths to be relative to the
project root (`models/model.pkl`) and deploy running from the root
instead of from inside `app/`.

---

## Recap: what was actually learned here

1. Recognizing and confirming severe class imbalance before doing
   anything else with the data
2. Why accuracy is actively misleading on imbalanced data — proven with
   real numbers, not just asserted
3. Stratified train/test splitting and why plain random splitting fails
   on rare classes
4. `class_weight="balanced"` and why it changes what the model
   optimizes for
5. Confusion matrices and the four outcomes underlying every
   classification metric
6. Precision, Recall, F1, and ROC-AUC — what each measures and why
   they trade off against each other
7. Saving both a model *and* its preprocessing (the scaler) together,
   since one without the other breaks in production
8. Being explicit about a deployed demo's real-world limitations

---

## Further reading

- scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- Precision/Recall visual explainer: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
- ROC-AUC explained: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
- Handling imbalanced datasets (deeper dive): https://imbalanced-learn.org/stable/
- Dataset source and background: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Streamlit docs: https://docs.streamlit.io
