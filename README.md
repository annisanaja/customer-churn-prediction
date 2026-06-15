# 🔋 Customer Churn Prediction: PowerCo
### BCG Data Science Job Simulation

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Overview

This project was completed as part of the **BCG Data Science Job Simulation**, where I acted as a data scientist advising PowerCo, a major European utility company facing significant customer churn in its SME (Small & Medium Enterprise) segment.

The core business question:

> *"Is price sensitivity the main reason SME customers are leaving PowerCo and can we predict who will churn next?"*

The answer, backed by data, was **no** and the findings led to a strategic recommendation to shift away from blanket discounting toward targeted, relationship-driven retention.

---

## 📂 Project Structure

```
powerco-churn-analysis/
│
├── data/                                         # Raw and processed datasets (not included)
│
├── notebooks/
│   └── Random Forest.ipynb                       # Model building, evaluation, predictions
│
├── output/
│   └── out_of_sample_data_with_predictions.csv   # Model predictions with churn probability
│
├── Executive Summary (Churn Analysis).pdf        # Business-ready summary of findings
└── README.md
```

---

## 🔍 Problem Statement

PowerCo serves over **15,000 SME customers** and is experiencing a churn rate of nearly **10%**. The hypothesis from the client was that customers were leaving due to better prices elsewhere and that a **20% discount** on energy prices would be the most effective retention lever.

This project was tasked with:
1. Validating or disproving the price sensitivity hypothesis
2. Identifying the true drivers of churn
3. Building a predictive model to flag at-risk customers
4. Recommending a data-driven retention strategy

---

## 🧪 Methodology

### 1. Exploratory Data Analysis
- Analysed churn distribution across customer segments, sales channels, contract origin, and tenure
- Identified that churn was concentrated among **low-consumption, high-margin** customers, the most commercially valuable accounts
- Visualised price differentials across off-peak, peak, and mid-peak periods (Dec vs Jan)
- Found no strong linear correlation between price variables and churn

### 2. Feature Engineering
Key features engineered from raw data:

| Feature | Description |
|--------|-------------|
| `offpeak_diff_dec_jan_energy` | Price change in off-peak energy (Dec → Jan) |
| `offpeak_diff_dec_jan_power` | Price change in off-peak power (Dec → Jan) |
| `peak_mid_peak_var_mean_diff` | Mean price spread between peak and mid-peak |
| `months_activ` | Months since contract activation |
| `months_to_end` | Months until contract expiry |
| `months_modif_prod` | Months since last product modification |
| `tenure` | Total months as a customer |

Additional preprocessing:
- **Log transformation** (`np.log10(x + 1)`) applied to skewed consumption variables
- **One-hot encoding** for `channel_sales` and `origin_up`, with rare categories (< 100 occurrences) dropped
- Rare dummy columns filtered to reduce noise

### 3. Correlation Analysis
- Full correlation heatmap generated across all features
- Identified multicollinearity between gross/net margin variables and consumption features
- Confirmed weak correlation between price-related features and churn

### 4. Random Forest Classifier
```python
model = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',   # handles class imbalance
    random_state=42
)
```

- Train/test split: **75% / 25%**, stratified by churn label
- Evaluation metric prioritised: **Recall** (minimising missed churners is costlier than false alarms)

---

## 📊 Model Results

| Metric | Score |
|--------|-------|
| Accuracy | 90.3% |
| Precision | 50.9% |
| Recall | 16.6% |
| F1 Score | 25.1% |

> ⚠️ **Note on accuracy:** A 90% accuracy baseline is achievable by predicting "no churn" for everyone, given the class imbalance. Recall is the more meaningful metric here, the model correctly flagged **59 out of 355 churners** in the test set. Further tuning with `class_weight` adjustments and threshold optimisation is recommended to improve Recall significantly.

### Churn Probability Output
```python
# Final output: customer-level churn predictions + probabilities
X_test['churn'] = predictions.tolist()
X_test['churn_probability'] = probabilities.tolist()
X_test.to_csv('out_of_sample_data_with_predictions.csv')
```
Customers are ranked by `churn_probability`, enabling the business to **prioritise the highest-risk accounts** for outreach.

---

## 💡 Key Findings

**1. Price is NOT the primary churn driver**
Price sensitivity analysis showed weak correlation between price changes and churn. Customers are not primarily leaving because of cost.

**2. Margin and consumption drive churn most**
Net and gross electricity margin, alongside 12-month consumption patterns, were the strongest predictors, suggesting that churn is linked to the customer's business performance or engagement level, not pricing.

**3. PowerCo is losing its best customers**
Churn is disproportionately concentrated among **low-consumption, high-margin** accounts, the most profitable segment. This makes the churn problem more costly than the raw 10% rate suggests.

**4. Discounting is the wrong lever**
A blanket 20% discount strategy would be expensive and largely ineffective given the findings. Price-insensitive customers won't be retained by price cuts.

---

## ✅ Recommendations

- **Shift from price-led to relationship-led retention**: invest in proactive customer outreach, particularly for high-margin accounts showing early churn signals
- **Target high-risk customers using the churn probability score**: prioritise accounts with `churn_probability > 0.6` for immediate engagement
- **Investigate why high-margin customers churn**: qualitative research (surveys, account manager interviews) is needed to understand the non-price drivers
- **Re-evaluate the discount hypothesis**: reserve discounts for price-sensitive segments where they are likely to have actual impact

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| pandas | Data manipulation and EDA |
| numpy | Numerical operations, log transformation |
| matplotlib / seaborn | Visualisation |
| scikit-learn | Random Forest, train/test split, metrics |

---

## 📁 Data

The dataset was provided as part of the BCG simulation and is not included in this repository for confidentiality reasons. It consisted of two primary tables:

- **client_df**: customer-level attributes (contract, channel, tenure, margin, consumption)
- **price_df**: monthly price data per customer across off-peak, peak, and mid-peak periods

---

## 👩‍💻 Author

**Naja Annisa Arifin**
Business & Data Analyst | Civil Engineering → Analytics
📍 Jakarta, Indonesia → Birmingham, UK (MSc Business Analytics, 2026)
🔗 [Portfolio](https://najaannisa.netlify.app) · [GitHub](https://github.com/annisanaja)

---

## 📄 License
This project is for educational and portfolio purposes as part of the BCG Data Science Job Simulation via Forage.
