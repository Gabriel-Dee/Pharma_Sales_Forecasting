# Streamlit Sales Dashboard with Shadcn UI

This project presents a sales dashboard utilizing Streamlit and the [Shadcn UI package](https://github.com/ObservedObserver/streamlit-shadcn-ui). It provides insights into sales data across different cities, showcasing both annual and monthly performance metrics.

## Healthcare Products Sales Forecasting in Tanzania

**Author:** Gabriel D. Minzemalulu  
**Date:** 21.06.2024

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding of the Challenge](#understanding-of-the-challenge)
3. [Solution Approach](#solution-approach)
   - [Data Exploration and Cleaning](#data-exploration-and-cleaning)
   - [Insights from Exploratory Data Analysis (EDA)](#insights-from-exploratory-data-analysis-eda)
   - [Model Development](#model-development)
     - [Univariate Analysis](#univariate-analysis)
     - [Multivariate Analysis](#multivariate-analysis)
   - [Model Evaluation](#model-evaluation)
   - [Visualization and Presentation](#visualization-and-presentation)
4. [Results](#results)
5. [Conclusion](#conclusion)

---

## Introduction 
This readme outlines the methodologies and findings from the Afya Intelligence Data Science Challenge, focusing on forecasting healthcare product sales in Tanzania over the next six months.

---

## Understanding of the Challenge 

The challenge required developing a forecasting model to predict future demand for healthcare products based on historical sales data. Key tasks included data exploration, model development, evaluation, and presentation of findings.

---

## Solution Approach 

### Data Exploration and Cleaning 

Initial steps involved thorough exploratory data analysis (EDA) using tools such as Pandas, Seaborn, Matplotlib, Scipy, and Statsmodels. Data cleaning included handling missing values, removing duplicates, outlier detection, and feature engineering.

### Insights from Exploratory Data Analysis (EDA) 
Insights gathered from EDA included sales trends over time, distribution of sales across pharmacies and products, and correlation analysis between variables.

### Model Development 

#### Univariate Analysis 

Several models were considered:
- **ARIMA**: RMSE: 397,508.36, MAE: 310,540.90, MAPE: 2.61
- **LSTM**: RMSE: 556,294.40, MAE: 397,475.76, MAPE: 3.38
- **Prophet**: RMSE: 488,721.44, MAE: 356,763.28, MAPE: 2.97

#### Multivariate Analysis

Models included:
- **VAR**: RMSE: 213,526.52, MAE: 165,947.17, MAPE: 279,873.57
- **ARIMAX**: RMSE: 431,895.61, MAE: 371,726.07, MAPE: 546,105.31
- **Prophet (Multivariable)**: RMSE: 465,903,914.82, MAE: 465,903,819.59, MAPE: 299,047,986.40

### Model Evaluation 

Models were evaluated using RMSE, MAE, and MAPE metrics to determine accuracy and reliability in forecasting.

### Visualization and Presentation 

Visualizations such as time series plots and forecast comparisons were used to effectively communicate findings and model performance.

---

## Results 

The VAR model emerged as the best performer, providing the most accurate forecasts for the next six months.

A CSV/Excel file containing the cleaned dataset and forecasted data was generated.

---

## Conclusion 

This challenge demonstrated effective application of time series forecasting models for healthcare demand prediction. The VAR model was recommended for its superior performance in accuracy metrics compared to other models evaluated.

**Recommendations:**
- **VAR Model**: Recommended based on lowest RMSE and MAE.
- **ARIMAX Model**: Reasonable performance but higher errors compared to VAR.
- **Prophet Model**: Performance significantly worse, indicating potential unsuitability for this dataset.

---

This readme file summarizes the approach, methodologies, and outcomes of the Afya Intelligence Data Science Challenge, offering insights into forecasting healthcare product sales in Tanzania.