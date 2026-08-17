# Marketing A/B Test & Conversion Prediction

A marketing analytics project combining **A/B testing** and **machine learning** to evaluate advertising effectiveness, predict user conversion behavior, handle severe class imbalance, and optimize advertising targeting strategies.

---

# Project Overview

Online advertising campaigns generate large amounts of user interaction data. However, conversion events are usually rare, creating challenges for both statistical analysis and predictive modeling.

This project aims to answer three key questions:

1. **Does advertising significantly improve conversion compared with PSA (Public Service Announcement)?**
2. **How do advertisement exposure patterns influence user conversion behavior?**
3. **Can machine learning models effectively identify potential converters under severe class imbalance?**

The project includes:

- Exploratory Data Analysis (EDA)
- A/B hypothesis testing
- User behavior analysis
- Logistic regression modeling
- Feature engineering
- Class imbalance handling
- Threshold optimization
- Business-oriented profit analysis

---

# Dataset

Dataset:

`marketing_AB.csv`

Dataset size:

- **588,101 users**
- **6 features**

Features:

| Feature | Description |
|---|---|
| user id | Unique user identifier |
| test group | Advertisement group (`ad`) or PSA group (`psa`) |
| converted | Whether the user completed conversion |
| total ads | Number of advertisements viewed |
| most ads day | Day with the highest advertisement exposure |
| most ads hour | Hour with the highest advertisement exposure |

---

# Exploratory Data Analysis

## Experiment Group Distribution

The experiment contains:

| Group | Users | Percentage |
|---|---:|---:|
| ad | 564,577 | 96% |
| psa | 23,524 | 4% |

The original distribution is preserved because it represents the real advertising experiment setting.

---

# A/B Testing Analysis

## Conversion Rate Comparison

Conversion results:

| Group | Conversions | Conversion Rate |
|---|---:|---:|
| ad | 14,423 | 2.55% |
| psa | 420 | 1.79% |

The advertisement group achieved:

**43.1% relative conversion improvement**

compared with the PSA group.

---

## Statistical Significance Test

A Chi-square test was performed.

Hypothesis:

- H0: Conversion rate is the same between ad and PSA groups
- H1: Conversion rate is different

Results:

```
Chi-square statistic = 54.01

p-value = 1.99e-13
```

Since:

```
p < 0.05
```

the null hypothesis is rejected.

### Conclusion

Advertisement exposure significantly improves conversion rate.

---

# User Behavior Analysis

## Advertisement Exposure Effect

Users were grouped according to advertisement exposure frequency.

| Ads Exposure | Conversion Rate |
|---|---:|
| 0-5 | 0.25% |
| 6-10 | 0.49% |
| 11-20 | 0.84% |
| 21-50 | 2.89% |
| 51-100 | 11.40% |
| 100+ | 16.91% |

A strong positive relationship exists between advertisement exposure and conversion probability.

However, this relationship represents correlation rather than guaranteed causal impact.

---

## Time Pattern Analysis

Conversion behavior varies by:

- Day of week
- Hour of day

Important patterns:

- Highest conversion day: **Monday**
- Highest conversion hour: **16:00**

These insights suggest potential opportunities for optimizing advertising schedules.

---

# Conversion Prediction Modeling

Because only around **2.5% of users converted**, the dataset suffers from severe class imbalance.

Therefore, accuracy alone is not an appropriate evaluation metric.

Models were evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Experiment Design

## Experiment 1
### Original Logistic Regression

Baseline model without imbalance handling.

Purpose:

Establish the initial prediction performance.

---

## Experiment 2
### Balanced Logistic Regression

Applied:

```
class_weight="balanced"
```

Purpose:

Increase the importance of minority class samples and improve conversion detection.

---

## Experiment 3
### Balanced Logistic Regression + log(total ads)

Applied logarithmic transformation:

```
log(1 + total ads)
```

Purpose:

Reduce the influence of extreme advertisement exposure values.

---

## Experiment 4
### Balanced Logistic Regression + Interaction Features

Added interaction features between:

- Advertisement group
- Advertisement exposure

Purpose:

Capture more complex relationships between advertising strategy and user behavior.

---

## Experiment 5
### Threshold Optimization

The default classification threshold (0.5) was optimized according to different business objectives.

Two strategies were explored:

1. Recall-oriented optimization
2. Profit-oriented optimization

---

# Model Performance Comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Original Logistic | 0.814 | 0.126 | 0.197 | 0.015 | 0.028 |
| Balanced Logistic | 0.855 | 0.135 | 0.118 | 0.715 | 0.202 |
| Balanced + log ads | 0.860 | 0.139 | 0.084 | 0.820 | 0.153 |
| Balanced + Interaction | 0.855 | 0.135 | 0.118 | 0.714 | 0.202 |

---

# Key Findings

## 1. Accuracy is misleading under class imbalance

The original model achieved:

```
Accuracy = 97.36%
```

However:

```
Recall = 1.52%
```

The model failed to identify most potential customers.

---

## 2. Class balancing significantly improves conversion detection

After applying class weighting:

Recall increased:

```
1.5% → 71.5%
```

ROC-AUC improved:

```
0.814 → 0.855
```

The model became much better at identifying potential converters.

---

## 3. Advertisement exposure is the strongest behavioral indicator

Feature analysis shows:

Top positive factors:

| Feature | Odds Ratio |
|---|---:|
| Ads viewed around 16:00 | 1.55 |
| Monday exposure | 1.46 |
| Tuesday exposure | 1.39 |
| Evening exposure hours | >1.20 |

These factors indicate higher conversion probability.

---

# Threshold Optimization Results

## Recall-oriented Strategy

Goal:

Capture more potential customers.

Selected threshold:

```
0.40
```

Results:

| Metric | Value |
|---|---:|
| Precision | 0.085 |
| Recall | 0.810 |
| F1-score | 0.155 |

This strategy increases customer coverage but creates more false positives.

---

## Profit-oriented Strategy

A business objective function was introduced:

```
Profit = Revenue - Advertising Cost
```

Best threshold:

```
0.32
```

Results:

| Metric | Value |
|---|---:|
| Targeted Users | 45,910 |
| Revenue | 26,440 |
| Cost | 4,591 |
| Profit | 21,849 |

This demonstrates how predictive models can be connected with business decisions.

---

# Project Structure

```
Marketing_AB_Test/

│
├── README.md
│
├── data/
│   └── marketing_AB.csv
│
├── notebooks/
│   ├── 01_EDA_and_AB_Test.ipynb
│   └── 02_Conversion_Prediction_Modeling.ipynb
│
├── figures/
│   ├── experiment_group_distribution.png
│   ├── conversion_rate_comparison.png
│   ├── ads_exposure_analysis.png
│   ├── conversion_by_day.png
│   ├── conversion_by_hour.png
│   └── group_exposure_interaction.png
│
└── requirements.txt
```

---

# How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks:

```
01_EDA_and_AB_Test.ipynb

02_Conversion_Prediction_Modeling.ipynb
```

---

# Future Improvements

Potential improvements include:

### Advanced Machine Learning Models

- XGBoost
- LightGBM
- Random Forest
- Gradient Boosting


### Causal Analysis

Further investigate whether advertisement exposure directly causes conversion:

- Propensity Score Matching
- Causal Forest
- Uplift Modeling


### Business Optimization

Improve targeting strategy using:

- Customer lifetime value
- Cost-sensitive learning
- Real-time bidding optimization

---

# Conclusion

This project demonstrates a complete workflow from **statistical experimentation to predictive modeling and business optimization**.

The analysis shows that:

- Advertising significantly improves conversion rates.
- Exposure patterns provide valuable behavioral insights.
- Handling class imbalance is essential for conversion prediction.
- Threshold optimization can translate machine learning predictions into practical marketing decisions.
