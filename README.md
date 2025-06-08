# House Price Prediction with Machine Learning
## Project Overview

This project aims to develop and evaluate machine learning models for accurate house price prediction using a real-world dataset. The goal is to identify a robust predictive model capable of handling the complexities and nuances inherent in real estate data.

## Key Features

* **Comprehensive Data Preprocessing:** Includes data cleaning, feature engineering, handling missing values, and crucial outlier detection and removal.
* **Target Variable Transformation:** Application of `log1p` transformation to address the skewed distribution of house prices.
* **Comparative Model Analysis:** Evaluation of two distinct machine learning algorithms:
    * **Ridge Regression:** A linear model with L2 regularization.
    * **XGBoost Regressor:** A powerful gradient boosting ensemble model.
* **Performance Evaluation:** Models are assessed using standard metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).
* **Jupyter Notebook:** All analysis, modeling, and evaluation steps are documented within a single, executable Jupyter Notebook.

## Methodology Highlights

1.  **Data Loading & Initial Cleaning:** Renaming columns, converting 'Amount' strings (e.g., "lac", "cr") to numeric values, and dropping initial irrelevant columns.
2.  **Extreme Outlier Removal:** Identification and removal of a significant outlier in the 'Total Amount' (e.g., a 14 Billion Rupee property identified as an error). This step proved critical for model stability.
3.  **Feature Engineering:** Cleaning and standardizing numerical features like 'Carpet Area', 'Floor', 'Bathroom Count', 'Balcony Count', and 'Car Parking Availability'.
4.  **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets.
5.  **Preprocessing Pipelines:**
    * Numerical features are imputed (median) and scaled (`StandardScaler`).
    * Categorical features are imputed (most frequent) and one-hot encoded (`OneHotEncoder`).
    * `ColumnTransformer` is used to apply these transformations systematically.
6.  **Model Training & Evaluation:** Both Ridge Regression and XGBoost Regressor are trained on the preprocessed data. Predictions are inverse-transformed to the original scale for evaluation against actual prices.

## Results & Key Findings

The evaluation clearly indicates the superior performance of the **XGBoost Regressor**.

| Model               | MAE (Rupees)      | RMSE (Rupees)      | R-squared (R²) |
| :------------------ | :---------------- | :----------------- | :------------- |
| Ridge Regression    | 3,290,660.16      | 44,093,369.53      | -8.0590        |
| **XGBoost Regressor** | **1,030,580.77** | **6,476,564.86** | **0.8046** |

XGBoost Regressor** achieved a strong R-squared of 0.8046, explaining over 80% of the variance in house prices, demonstrating its effectiveness in capturing complex non-linear relationships.
* **Ridge Regression** performed poorly with a negative R-squared, indicating its unsuitability for this dataset's inherent non-linear patterns, even after extensive preprocessing.
* The **removal of the extreme outlier** was crucial, significantly improving the stability and meaningfulness of model evaluation metrics for both algorithms.
