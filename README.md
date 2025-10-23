# Proactive Delivery Delay Mitigation Strategy

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-2.0-blue.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-blue.svg)](https://xgboost.ai/)

This end-to-end data science project moves from raw, multi-table e-commerce data to a complete, ROI-positive business strategy. The goal is to not only *predict* delivery delays but also to identify the most *profitable* intervention strategy to mitigate them.

This project was built to demonstrate the end-to-end lifecycle of a Business Analyst, from data extraction and feature engineering to predictive modeling, root-cause analysis, and a final, profitable strategic recommendation.

## 🚀 The Core Business Problem

For a major e-commerce platform like Flipkart, delivery delays are a primary driver of customer dissatisfaction, increased operational costs (refunds, compensation vouchers), and a lower Net Promoter Score (NPS).

This project answers two critical questions:
1.  **Prediction:** Which orders are at the highest risk of missing their promised delivery date?
2.  **Prescription:** What are the root causes of these delays, and what is the most *profitable* way to intervene (e.g., upgrade shipping) to prevent them *before* they happen?

## 📊 Methodology: The 7-Step Analytics Pipeline

This project is broken down into a single notebook (`delivery_delay_prediction.ipynb`) that follows a complete BA workflow.

### Step 1: Data Ingestion & Cleaning
* Loaded 8 raw `.csv` files (e.g., `orders`, `customers`, `order_items`, `sellers`, etc.) into Pandas DataFrames.
* Merged all tables into a single master DataFrame, handling duplicates and cleaning data types.

### Step 2: Problem Framing & Feature Engineering
* Defined the business problem by creating two target variables:
    1.  `delayed_flag` (Binary 1/0): The target for our classification model.
    2.  `delivery_delay_days` (Continuous): The target for our regression model.
* Engineered 10+ new, high-value features from the raw data:
    * `seller_to_customer_distance_km`: Calculated using the **Haversine formula** to get the real-world distance from seller to customer coordinates.
    * `seller_on_time_rate`: A powerful custom feature quantifying a seller's *historical* on-time delivery performance.
    * `estimated_shipping_time_days`: The delivery SLA (in days) promised to the customer.

### Step 3: Exploratory & Diagnostic Analysis
* Established the baseline business metric: **6.77% of all delivered orders are delayed**.
* **Crucial Insight:** Proved that **raw distance is a poor predictor** of delays; many short-distance orders were severely delayed, while long-distance ones were early, pointing to operational bottlenecks.
* **Key Finding:** Seller performance (our `seller_on_time_rate` feature) and product category were much stronger drivers, confirming the problem was operational, not just logistical.

### Step 4: Predictive Modeling & Model Selection
* **The Bake-Off:** Built two models to predict `delayed_flag`: a simple **Logistic Regression** and an advanced **XGBoost Classifier**.
* **Crucial Discovery (The Pivot):** The "advanced" XGBoost model performed poorly (PR-AUC 0.09) and was unstable. I diagnosed this as **overfitting to 'concept drift'**—the real-world reasons for delays were changing over time, and the complex model couldn't adapt.
* **The Decision:** Selected the **simpler, more stable Logistic Regression** model as the champion. It was more robust and provided a reliable predictive signal (PR-AUC 0.1703).
* **Failed Model Insight:** Also proved that a *regression model* (to predict the *exact number of days* delayed) was **not feasible**, performing worse than a naive baseline. This saves the company from investing in a non-viable project.

### Step 5: Model Explainability (Root-Cause Attribution)
* Extracted the coefficients from the winning Logistic Regression model to find the *exact* drivers of delay.
* **Key Insight:** Geography is the #1 driver. The top predictor of a delay was a customer being in state `AL`.
* **Key Insight:** Discovered the **"São Paulo Paradox"**: *Sellers* in São Paulo (`seller_state_SP`) were a **major cause of delays**, but *Customers* in São Paulo (`customer_state_SP`) were in the **safest delivery zone**. This pinpoints the bottleneck to seller-side operations in SP, not last-mile delivery.

### Step 6: ROI Simulation & Policy Iteration
* This is the "money" step. I built a profit simulator to find a profitable business policy.
* **Assumptions:**
    * `Avg. Compensation Cost per Delay`: $25 (e.g., voucher, refund)
    * `Cost Per Intervention`: $5 (e.g., upgrade shipping)
* **The Initial Failure:** My first simulation, targeting the "Top 5% high-risk orders," was **unprofitable** and **resulted in a $575 loss** because it was too imprecise (cost of false positives was too high).
* **The Strategy & Win:** I iterated on the policy and tested a more precise, "surgical" approach targeting only the **"Top 2% highest-risk orders."**
* **Final Result:** This new, data-driven policy was profitable, generating **$445 in net savings** on the test set.

### Step 7: Final Dashboard & Recommendation
* Created the key visuals for a final stakeholder dashboard.
    1.  **"Early Warning List"**: A table of the Top 10 at-risk orders for the operations team to act on.
    2.  **"ROI Simulation"**: A bar chart clearly showing the cost of the "Baseline" ($16,850) vs. the "With Our Model" ($16,405) scenarios.

## 🔑 Key Insights
1.  **Operational Factors > Logistical Factors:** The *seller's* performance and location (`seller_on_time_rate`, `seller_state_SP`) were far stronger predictors of delay than the *customer's* location or distance. The problem is in the warehouse, not on the road.
2.  **Simple & Stable > Complex & Unstable:** A simple Logistic Regression model was the correct business choice. It **outperformed the complex XGBoost** model because it was more robust to real-world "concept drift" in the data, proving that the most complex model is not always the best one.
3.  **Profitability is in the Policy, Not the Model:** The model itself doesn't make money. A naive "Top 5%" intervention policy *lost money*. The real value was found by **iterating on the policy** to a "Top 2%" surgical approach, which turned a **$575 loss into a $445 profit.**

## 🛠️ How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/e-commerce-delivery-prediction.git
    cd e-commerce-delivery-prediction
    ```
2.  **Set up the environment:**
    * Upload the `delivery_delay_prediction.ipynb` notebook to Google Colab.
    * The notebook's first cell will install all required libraries (`kaggle`, `xgboost`, etc.).
3.  **Get Kaggle API Key:**
    * Follow the instructions in the notebook (Step 0.3.1) to download your `kaggle.json` API key.
4.  **Run the Notebook:**
    * Run the cells in order. The notebook will automatically download the data, perform all analysis, and build all models.

## 📦 Libraries Used
* `pandas`
* `numpy`
* `matplotlib` / `seaborn`
* `scikit-learn` (for data processing, Logistic Regression, and Pipelines)
* `xgboost` (for modeling and comparison)
* `kaggle` (for data ingestion)
