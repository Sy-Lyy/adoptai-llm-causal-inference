# AdoptAI — LLM Decision Support via Causal Inference

Predicting cat adoption outcomes using causal inference (logistic regression + PSM) and generating actionable shelter recommendations via a structured HuggingFace LLM prompt.

---

## Overview

Animal shelters struggle to understand which cats are less likely to be adopted and how to address that. This project uses Petfinder API data to identify the key drivers of adoption, quantifies the causal effect of sociability using Propensity Score Matching, and feeds those findings into an LLM to generate practical shelter strategies.

---

## Results

**Feature correlations with adoption status:**

| Feature | Correlation |
|---|---|
| `age_encoded` | −0.19 |
| `house_trained` | +0.18 |
| `unknown_count` | −0.14 |
| `good_with_children` | +0.13 |
| `good_with_cats` | +0.10 |
| `special_needs` | −0.07 |

**PSM Result:** Highly sociable cats (top 30% by sociability score) are **6.2 percentage points more likely to be adopted** (ATT = 0.062), after controlling for age, health status, and missing information.

---

## Pipeline

```
Petfinder API → data collection → cleaning → EDA → logistic regression + PSM → LLM recommendations
```

---

## Repo Structure

```
├── README.md
├── assets/
├── notebooks/
│   └── cat_adoption_llm_notebook.ipynb
└── requirements.txt
```

---

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn · HuggingFace Transformers (`openchat/openchat-3.5-1210`) · Petfinder API

---

## Note on Data

Data is fetched live from the Petfinder API and not included in this repository.
