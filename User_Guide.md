## 1. Overview

The application is an interactive web application that predicts customer churn, quantifies financial 
exposure, and recommends budget-optimised retention strategies. The system accepts a customer CSV 
file and produces per-customer churn probability scores, CLV-based financial metrics, scenario-driven 
budget allocation, and SHAP-based churn driver explanations — all within a single dashboard.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2. Requirements

- Python 3.11 or above
- All dependencies listed in `requirements.txt`
- A CSV file containing customer data (see FYP Datasets)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 3. Launching the Application

  1. Clone or download the Application repository.

  2. Open a terminal window in the project directory and install dependencies: pip install -r requirements.txt

  3. Launch the application using terminal: streamlit run app.py

  4. The application will open automatically in your default browser.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 4. Preparing Your Data

The system accepts any CSV file containing the following required columns, 
which can be obtained from FYP Datasets:
| Column | Description |
| `customerID` | Unique customer identifier |
| `tenure` | Number of months the customer has been with the operator |
| `MonthlyCharges` | Current monthly billing amount |
| `TotalCharges` | Total charges accumulated over the customer's lifetime |
| `Churn` | Churn label — accepts `Yes/No`, `1/0`, `True/False`, `churned/active` |

Additional columns (e.g., contract type, internet service, tech support) are automatically detected and incorporated as model features.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 5. Using the Dashboard

### 5.1 Sidebar Controls
All system controls are located in the left sidebar and update the dashboard in real time without retraining.

| Control | Description |
| **Model Selection** | Toggle between Random Forest (higher performance) and Logistic Regression (higher interpretability) |
| **High Risk Threshold** | Churn probability above which a customer is classified as high risk (default: 0.70) |
| **Medium Risk Threshold** | Churn probability above which a customer is classified as medium risk (default: 0.40) |
| **Retention Budget ($)** | Total retention spend available for allocation across customer segments |
| **Display Options** | Displays model functions such as high-risk table, financial dashboard, and SHAP explanations (default: all ticked) |

### 5.2 Uploading Data
Click **Browse files** in the Upload panel and select your prepared CSV. The system will:
- Validate that all required columns are present — an error is displayed if any are missing
- Display a data preview for confirmation
- Automatically run the full preprocessing and modelling pipeline upon successful validation

### 5.3 Churn Prediction Results

Once data is uploaded, the Churn Prediction panel displays:
- **KPI summary** — total customers, high/medium/low risk counts and percentages
- **Churn probability distribution** — binned bar chart showing how probability scores are distributed across the customer base
- **High-risk customer table** — all customers above the high-risk threshold, sorted by churn probability, showing customerID, probability score, risk segment, monthly charges, total charges, and actual churn label where available
- **Download button** — exports the full prediction results as a CSV file containing customerID, churn probability, risk segment, tenure, monthly charges, and total charges

### 5.4 Financial Impact Dashboard

The Financial Impact panel converts churn probability scores into revenue metrics:

- **Total CLV** — sum of Customer Lifetime Value (MonthlyCharges × tenure) across all customers
- **CLV at Risk** — total CLV concentrated in the high-risk cohort, expressed as a dollar amount and percentage of total CLV
- **Average CLV of Churners** — mean CLV among customers labelled as churned in the uploaded dataset

### 5.5 Retention Strategy Panel

**Customer Intelligence Table.** Displays each retention tier (Priority, Standard, Passive) with customer count, average churn probability, average CLV, total Value at Risk, budget weight, budget allocated, and cost per customer.

- **Priority** — high churn risk AND above-median CLV; highest ROI target
- **Standard** — high churn risk OR above-median CLV
- **Passive** — low churn risk AND below-median CLV

**Scenario Selection.** Choose from four retention scenarios using the dropdown:

| Scenario | Best Used When |
| Balanced Approach | Moderate churn risk, mid-range budget |
| Cost-Efficient | Large at-risk base, tight budget |
| High-Value Customer Focus | Small high-CLV segment, generous budget |
| Automation-First | Any budget, minimal manual intervention preferred |

**Priority Customer List.** Displays the top N priority customers ranked by Value at Risk (ChurnProbability × CLV). Use the slider to adjust how many customers are shown.

**Strategy Budget Breakdown.** Shows how the Priority tier budget is distributed across specific intervention types for the selected scenario. Toggle between horizontal bar and donut chart views.

**ROI Calculator.** Adjust the assumed retention success rate slider to model different campaign outcomes. The calculator displays expected CLV saved, total budget spent, net benefit, and ROI percentage. A green confirmation appears when the campaign is projected to be net-positive; an amber warning appears when the budget does not cover expected returns at the assumed success rate.

### 5.6 SHAP Interpretability Panel

The Churn Drivers panel explains what is driving churn across the high-risk customer cohort:

- **Top factors bar chart** — top 10 features ranked by mean absolute SHAP value, reflecting overall influence on churn probability
- **Churn driver table** — features with positive mean SHAP values only, actively pushing customers toward churn, with impact strength and business insight for each driver
- **Executive summary** — plain-language summary of the top three churn drivers with recommended intervention focus areas

### 5.7 Model Evaluation Panel

The Evaluation panel displays model performance metrics computed on the held-out test set:

- **Accuracy, Precision, Recall, F1 Score, AUC-ROC** — displayed as KPI cards
- **Confusion matrix** — interactive heatmap showing true positives, true negatives, false positives, and false negatives
- **ROC curve** — model discrimination curve plotted against the random baseline diagonal
- **Feature importance chart** — top 15 features by Gini impurity reduction (Random Forest) or absolute coefficient magnitude (Logistic Regression)

For Logistic Regression, a 5-fold cross-validated AUC is also displayed, providing a more statistically reliable performance estimate than the single-split metrics.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 6. Recommended Testing Walkthrough

For moderator verification, the following sequence is recommended:

1. Launch the application following Section 3
2. Upload the WA Telco dataset (available at: https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
3. Confirm the data preview and success message appear correctly
4. Set model to **Random Forest**, thresholds at defaults (0.70 / 0.40), budget at **$5,000**
5. Review the KPI summary and churn probability distribution
6. Scroll to the Financial Impact panel and verify CLV metrics are populated
7. Select the **High-Value Customer Focus** scenario and review the tier intelligence table
8. Set the ROI success rate to **25%** and confirm a net-positive projection appears
9. Scroll to the SHAP panel and verify the churn driver table and executive summary populate
10. Switch model to **Logistic Regression** and confirm all panels update consistently
11. Download the predictions CSV and verify the file contains the correct columns and format

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 7. Troubleshooting

| Issue | Resolution |
| `Missing required columns` error on upload | Ensure your CSV contains all five required columns with exact spelling as listed in Section 4 |
| SHAP computation takes long | Expected on large datasets; capped at 200 samples automatically — allow up to 30 seconds |
| Logistic Regression convergence warning | Increase `max_iter` in the code from 1000 to 2000 for datasets with very high feature dimensionality |
| `TotalCharges` column shows errors | Ensure TotalCharges contains numeric values; whitespace entries are handled automatically but fully non-numeric columns will fail validation |
| Application does not open in browser | Navigate manually to `http://localhost:8501` in your browser |
