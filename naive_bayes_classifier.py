#!/usr/bin/env python3
"""
CSC510 Module 6 - Critical Thinking: Naive Bayes Classifier (scikit-learn)

Model: CategoricalNB (a Multinomial-style Naive Bayes for categorical features)
Dataset: classic "Play Golf" weather data (14 rows, 4 categorical features).

What the script does
  1. Converts the dataset into FREQUENCY tables (feature value x class counts).
  2. Builds LIKELIHOOD tables P(feature value | class), without and with
     Laplacian (add-one) correction, showing the zero-probability problem.
  3. Calculates the POSTERIOR probability for each class by hand using Bayes' Theorem:
         P(class | x) = P(class) * prod_i P(x_i | class) / P(x)
  4. Trains scikit-learn's CategoricalNB(alpha=1.0) and checks that it
     reproduces the hand-calculated posteriors.

Usage
  python naive_bayes_classifier.py                       # default demo sample
  python naive_bayes_classifier.py Outlook=Sunny Temperature=Cool Humidity=High Wind=Strong
  python naive_bayes_classifier.py Outlook=Overcast Humidity=High   # unspecified features are skipped
"""

import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

ALPHA = 1.0  # Laplace (add-one) smoothing constant
TARGET = "Play"

# --------------------------------------------------------------------------- #
# 1. Dataset
# --------------------------------------------------------------------------- #
DATA = pd.DataFrame(
    [
        ("Sunny",    "Hot",  "High",   "Weak",   "No"),
        ("Sunny",    "Hot",  "High",   "Strong", "No"),
        ("Overcast", "Hot",  "High",   "Weak",   "Yes"),
        ("Rainy",    "Mild", "High",   "Weak",   "Yes"),
        ("Rainy",    "Cool", "Normal", "Weak",   "Yes"),
        ("Rainy",    "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"),
        ("Sunny",    "Mild", "High",   "Weak",   "No"),
        ("Sunny",    "Cool", "Normal", "Weak",   "Yes"),
        ("Rainy",    "Mild", "Normal", "Weak",   "Yes"),
        ("Sunny",    "Mild", "Normal", "Strong", "Yes"),
        ("Overcast", "Mild", "High",   "Strong", "Yes"),
        ("Overcast", "Hot",  "Normal", "Weak",   "Yes"),
        ("Rainy",    "Mild", "High",   "Strong", "No"),
    ],
    columns=["Outlook", "Temperature", "Humidity", "Wind", TARGET],
)
FEATURES = [c for c in DATA.columns if c != TARGET]


# --------------------------------------------------------------------------- #
# 2. Frequency + likelihood tables
# --------------------------------------------------------------------------- #
def frequency_tables(df):
    """Return {feature: DataFrame of counts (rows=feature value, cols=class)}."""
    return {f: pd.crosstab(df[f], df[TARGET]) for f in FEATURES}


def likelihood_tables(freq, class_counts, alpha=0.0):
    """
    P(feature value | class) = (count + alpha) / (class_count + alpha * n_values)
    alpha=0 -> raw maximum-likelihood estimate (can contain zeros)
    alpha=1 -> Laplacian (add-one) correction
    """
    out = {}
    for f, table in freq.items():
        n_values = table.shape[0]
        out[f] = (table + alpha) / (class_counts + alpha * n_values)
    return out


def manual_posterior(sample, priors, likelihoods):
    """Bayes' Theorem with the naive independence assumption, normalised over classes."""
    unnormalised = {}
    for cls in priors.index:
        p = priors[cls]
        for feat, val in sample.items():
            # An unseen value gets add-one probability when smoothing is on
            table = likelihoods[feat]
            p *= table.loc[val, cls] if val in table.index else 0.0
        unnormalised[cls] = p
    total = sum(unnormalised.values())
    if total == 0:  # every class zeroed out (only possible without smoothing)
        return unnormalised, {c: float("nan") for c in unnormalised}
    return unnormalised, {c: v / total for c, v in unnormalised.items()}


# --------------------------------------------------------------------------- #
# 3. scikit-learn model
# --------------------------------------------------------------------------- #
def train_sklearn(df, alpha=ALPHA):
    enc = OrdinalEncoder()
    X = enc.fit_transform(df[FEATURES]).astype(int)
    model = CategoricalNB(alpha=alpha)  # alpha=1.0 => Laplace smoothing
    model.fit(X, df[TARGET])
    return enc, model


def sklearn_posterior(enc, model, sample):
    row = pd.DataFrame([sample], columns=FEATURES)
    probs = model.predict_proba(enc.transform(row).astype(int))[0]
    return dict(zip(model.classes_, probs))


# --------------------------------------------------------------------------- #
# 4. Helpers
# --------------------------------------------------------------------------- #
def parse_sample(argv):
    """Parse Feature=Value arguments; fall back to a default demo sample."""
    if not argv:
        return {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
    sample = {}
    for arg in argv:
        if "=" not in arg:
            sys.exit(f"Bad argument '{arg}'. Use Feature=Value, e.g. Outlook=Sunny")
        k, v = arg.split("=", 1)
        if k not in FEATURES:
            sys.exit(f"Unknown feature '{k}'. Choose from: {', '.join(FEATURES)}")
        if v not in DATA[k].unique():
            sys.exit(f"Unknown value '{v}' for {k}. Choose from: {', '.join(sorted(DATA[k].unique()))}")
        sample[k] = v
    return sample


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    sample = parse_sample(sys.argv[1:])

    header("DATASET")
    print(DATA.to_string(index=False))

    # --- Frequency tables --------------------------------------------------
    class_counts = DATA[TARGET].value_counts().sort_index()
    priors = class_counts / class_counts.sum()
    freq = frequency_tables(DATA)

    header("STEP 1: FREQUENCY TABLES (counts)")
    print("Class counts:", class_counts.to_dict(), "\n")
    for f, t in freq.items():
        print(f"--- {f} ---\n{t}\n")
    print("Prior P(class):", {c: round(p, 4) for c, p in priors.items()})

    # --- Likelihood tables -------------------------------------------------
    raw = likelihood_tables(freq, class_counts, alpha=0.0)
    smooth = likelihood_tables(freq, class_counts, alpha=ALPHA)

    header("STEP 2: LIKELIHOOD TABLES  P(value | class)")
    for f in FEATURES:
        print(f"--- {f}: no smoothing ---\n{raw[f].round(4)}\n")
        print(f"--- {f}: Laplace (alpha={ALPHA:g}) ---\n{smooth[f].round(4)}\n")

    zeros = [(f, v, c) for f, t in raw.items() for v in t.index for c in t.columns if t.loc[v, c] == 0]
    print("Zero-probability entries without smoothing:", zeros or "none")
    print("Laplace fix: (count + 1) / (class_count + number_of_values) -> no entry is ever 0.")

    # --- Posterior ---------------------------------------------------------
    header("STEP 3: POSTERIOR FOR INPUT SAMPLE")
    print("Input:", sample)
    _, post_raw = manual_posterior(sample, priors, raw)
    unnorm, post_smooth = manual_posterior(sample, priors, smooth)

    print("\nUnnormalised score  P(class) * prod P(x_i | class)  [Laplace]:")
    for c, v in unnorm.items():
        print(f"  {c:>3}: {v:.6f}")

    # --- scikit-learn ------------------------------------------------------
    enc, model = train_sklearn(DATA)
    sk = sklearn_posterior(enc, model, {k: sample.get(k, None) for k in FEATURES}) if len(sample) == len(FEATURES) else None

    header("STEP 4: PREDICTION")
    if sk is None:
        print("(Partial input: scikit-learn needs all 4 features, so only the hand-computed "
              "Laplace posterior is shown.)\n")
    print(f"{'Class':<8}{'Manual (no smoothing)':>24}{'Manual (Laplace)':>20}"
          + (f"{'scikit-learn':>16}" if sk else ""))
    for c in priors.index:
        line = f"{c:<8}{post_raw[c]:>24.4f}{post_smooth[c]:>20.4f}"
        if sk:
            line += f"{sk[c]:>16.4f}"
        print(line)

    best = max(post_smooth, key=post_smooth.get)
    print(f"\nPredicted class: {best}  (probability {post_smooth[best]:.2%})")

    if sk:
        match = all(np.isclose(sk[c], post_smooth[c]) for c in sk)
        print(f"Manual Laplace posterior matches scikit-learn: {match}")
        acc = model.score(enc.transform(DATA[FEATURES]).astype(int), DATA[TARGET])
        print(f"Training accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()