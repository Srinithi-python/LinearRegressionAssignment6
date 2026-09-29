# Mobile Price Prediction using Linear Regression

## Project Overview

This project uses **Linear Regression** to predict mobile phone prices
based on selected technical features from a mobile price dataset.

The analysis covers: - Dataset exploration - Missing-value checking -
Statistical summaries - Correlation analysis - Feature selection -
Train/test splitting - Linear Regression model training - Price
prediction - Model evaluation - Insights and possible improvements

## Dataset

The dataset contains **161 rows and 14 columns**.

Dataset source:

https://raw.githubusercontent.com/Srinithi-python/LinearRegressionAssignment6/refs/heads/main/mobile_price.csv

### Dataset Columns

-   `Product_id`
-   `Price` --- target variable
-   `Sale`
-   `weight`
-   `resoloution`
-   `ppi`
-   `cpu core`
-   `cpu freq`
-   `internal mem`
-   `ram`
-   `RearCam`
-   `Front_Cam`
-   `battery`
-   `thickness`

The dataset contains **no missing values**.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

## Project Workflow

### 1. Explore the Dataset

The dataset was loaded using Pandas. The following checks were
performed:

-   First and last records
-   Dataset shape
-   Column names
-   Data types
-   Dataset information
-   Missing-value counts
-   Statistical summaries for numerical features

The dataset has:

-   **161 records**
-   **14 columns**
-   **0 missing values**

## 2. Correlation Analysis

A correlation matrix and heatmap were used to understand the
relationship between numerical features and `Price`.

The four features with the strongest absolute correlations with Price
were selected:

  Feature             Correlation with Price
  ----------------- ------------------------
  RAM                                 0.8969
  PPI                                 0.8176
  Internal Memory                     0.7767
  Rear Camera                         0.7395

### Correlation Insights

-   **RAM** has the strongest positive relationship with Price.
-   **PPI** has a strong positive relationship with Price.
-   **Internal Memory** has a strong positive relationship with Price.
-   **Rear Camera** also has a positive relationship with Price.

Scatter plots with regression lines were used to visually examine these
relationships. All four selected features show an overall upward trend
with Price.

> Correlation indicates association and does not by itself establish
> causation.

## 3. Feature Selection

### Input Features

The model uses:

``` python
features = ["ram", "ppi", "internal mem", "RearCam"]
```

### Target Variable

``` python
y = df["Price"]
```

The feature matrix is:

``` python
X = df[features]
```

Therefore:

-   **X:** RAM, PPI, Internal Memory, Rear Camera
-   **y:** Price

## 4. Train/Test Split

The dataset was divided into:

-   **80% training data**
-   **20% testing data**

The split used `random_state=42` for reproducibility.

Result:

-   Training set: **128 rows**
-   Testing set: **33 rows**

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

## 5. Linear Regression Model

A Linear Regression model from Scikit-learn was created and trained
using the training dataset.

``` python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

The trained model was then used to predict prices for the test dataset:

``` python
y_pred = model.predict(X_test)
```

## 6. Regression Coefficients and Intercept

The trained model produced the following coefficients:

  Feature             Coefficient
  ----------------- -------------
  RAM                    275.4658
  PPI                      1.4098
  Internal Memory          0.5495
  Rear Camera             19.9557

**Intercept:** `915.6007`

The coefficients represent the estimated change in predicted Price for a
one-unit increase in each feature while the other selected features are
held constant.

The resulting model can be represented conceptually as:

``` text
Price = 915.6007
        + 275.4658 × RAM
        + 1.4098 × PPI
        + 0.5495 × Internal Memory
        + 19.9557 × Rear Camera
```

## 7. Model Performance

The model was evaluated using R², MAE, MSE, and RMSE on the test
dataset.

  Metric          Result
  ---------- -----------
  R² Score        0.8528
  MAE             246.33
  MSE          83,466.34
  RMSE            288.91

### Metric Interpretation

**R² Score --- 0.8528**

The model explains approximately **85.28% of the variation in mobile
Price** in the test dataset.

**MAE --- 246.33**

The predictions differ from the actual prices by approximately **246.33
price units on average**, based on absolute error.

**MSE --- 83,466.34**

MSE represents the average squared prediction error. Larger errors have
a greater effect because the errors are squared.

**RMSE --- 288.91**

RMSE represents the typical size of prediction error in the same units
as the Price variable.

## 8. Key Insights

1.  **RAM is the strongest selected feature** based on its correlation
    with Price, with a correlation of approximately 0.90.
2.  **PPI, Internal Memory, and Rear Camera** also show strong positive
    associations with Price.
3.  The scatter plots show upward trends between the selected features
    and mobile Price.
4.  The Linear Regression model explains approximately **85.28% of the
    variation** in Price on the test data.
5.  The model still has prediction errors, which indicates that factors
    beyond the four selected features may influence mobile prices.

## 9. Areas for Improvement

Potential improvements include:

-   Adding additional relevant features such as battery, processor
    specifications, display characteristics, and other camera/storage
    specifications.
-   Performing additional feature engineering.
-   Investigating and handling outliers.
-   Checking for multicollinearity between independent variables.
-   Applying feature scaling where appropriate.
-   Using cross-validation to assess model stability.
-   Comparing Linear Regression with other regression algorithms such
    as:
    -   Ridge Regression
    -   Lasso Regression
    -   Decision Tree Regression
    -   Random Forest Regression
    -   Gradient Boosting Regression
-   Performing hyperparameter tuning for suitable models.

## 10. Conclusion

This project demonstrates how Linear Regression can be used to predict
mobile phone prices from technical specifications.

The selected features --- **RAM, PPI, Internal Memory, and Rear Camera**
--- have positive relationships with Price. Using an 80/20 train-test
split, the Linear Regression model achieved an **R² score of 0.8528**,
with an **MAE of 246.33** and **RMSE of 288.91** on the test data.

The results show that the selected features provide useful information
for predicting mobile prices, while additional features and alternative
machine-learning models could be explored to further improve prediction
performance.

## Project File

-   `LinearRegressionAss6.ipynb` --- Jupyter Notebook containing the
    complete analysis and model implementation.
