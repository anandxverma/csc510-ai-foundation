# Module 6 Critical Thinking: Naive Bayes Classifier

## Files
- `naive_bayes_classifier.py` – executable script (scikit-learn `CategoricalNB`)

## Setup
Requires Python 3.9+. Install the dependencies:

```bash
pip3 install scikit-learn pandas numpy
```

## Testing
Run the script from the folder containing it.

**Example 1 – default test sample:**

```bash
python3 naive_bayes_classifier.py
```

**Example 2 – full custom sample (all four features):**

```bash
python3 naive_bayes_classifier.py Outlook=Overcast Temperature=Cool Humidity=High Wind=Strong
```

**Example 3 – partial input (hand-computed posterior only):**

```bash
python3 naive_bayes_classifier.py Outlook=Overcast
```

Valid values: Outlook (Sunny/Overcast/Rainy), Temperature (Hot/Mild/Cool), Humidity (High/Normal), Wind (Weak/Strong).

Expected result for the default sample (Sunny, Cool, High, Strong): **Play = No, 72.01%** (Yes = 27.99%), and the script prints `Manual Laplace posterior matches scikit-learn: True`.

## What the script outputs
1. **Frequency tables** – counts of each feature value per class (`pd.crosstab`).
2. **Likelihood tables** P(value | class), with and without Laplace smoothing.
3. **Posterior** for each class via Bayes' Theorem: P(class | x) = P(class) · ∏ P(xᵢ | class) / P(x).
4. **Prediction** – predicted class plus probabilities from the manual calculation (with and without smoothing) next to scikit-learn's `predict_proba`.

## Why Laplacian correction matters
In the data, Outlook = Overcast never occurs with Play = No, so the raw likelihood is 0/5 = 0. Any sample with Overcast would then get a posterior of exactly 0 for "No" regardless of the other features (see the "no smoothing" column when you run the Overcast test). Add-one smoothing uses

P(value | class) = (count + 1) / (class_count + number_of_values)

so Overcast|No becomes 1/8 = 0.125, and no class can be eliminated by a single unseen combination. In scikit-learn this is `CategoricalNB(alpha=1.0)`.

## Why CategoricalNB
All four predictors are categorical (discrete labels with no numeric meaning), so I used `CategoricalNB`:
- **GaussianNB** assumes continuous, normally distributed features, which does not fit labels like "Sunny" or "Weak".
- **BernoulliNB** models binary (present/absent) features; it would need every category one-hot encoded, losing the natural multi-valued structure (Outlook has 3 values).
- **MultinomialNB** models counts (e.g., word frequencies). `CategoricalNB` is its direct equivalent for categorical features: it learns a separate multinomial distribution P(value | class) per feature, which is exactly the frequency/likelihood table approach used here.

## Naive independence assumption
The model treats features as conditionally independent given the class, so the joint likelihood is a simple product of per-feature likelihoods. This is rarely exactly true (e.g., Outlook and Humidity are correlated), but it keeps the model needing few parameters, works well on small data, and typically still ranks classes correctly even when the exact probabilities are somewhat off. The training accuracy of 92.86% (13/14) reflects this.