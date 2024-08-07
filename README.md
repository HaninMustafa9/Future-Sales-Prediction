# Future Sales Prediction

## Project Overview

This project aims to predict future sales based on advertising costs across different media channels: TV, Radio, and Newspaper. By using a linear regression model, the project explores how advertising expenditures impact sales and makes predictions for future sales.

## Data Preprocessing

### Import Libraries

The following libraries are used for data manipulation, visualization, and machine learning:
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical operations
- **matplotlib**: For creating static, animated, and interactive visualizations
- **seaborn**: For statistical data visualization
- **sklearn**: For machine learning and model evaluation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

### Load Data

The dataset is loaded from an external CSV file containing information about advertising costs and corresponding sales figures:

```python
Sales_Data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
```

### Exploratory Data Analysis (EDA)

1. **Descriptive Statistics**
   - View initial and final rows of the dataset
   - Examine dataset shape, data types, and summary statistics

```python
Sales_Data.head()
Sales_Data.tail()
Sales_Data.shape
Sales_Data.info()
Sales_Data.describe()
```

2. **Visualizations**
   - Histograms and boxplots to understand distributions and detect outliers
   - Scatter plots to explore relationships between advertising costs and sales

```python
fig, axes = plt.subplots(4, 2, figsize=(15, 25))
sns.histplot(Sales_Data['TV'], kde=True, ax=axes[0, 0])
sns.boxplot(x=Sales_Data['TV'], ax=axes[0, 1])
# (Similar plots for Radio, Newspaper, and Sales)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Scatter plots for TV, Radio, Newspaper vs Sales
plt.tight_layout()
plt.show()
```

### Data Cleaning

**Removing Outliers**
- Outliers in the Newspaper feature are identified and removed based on the interquartile range (IQR) to improve model performance.

```python
Q1_Newspaper = Sales_Data['Newspaper'].quantile(0.25)
Q3_Newspaper = Sales_Data['Newspaper'].quantile(0.75)
IQR_Newspaper = Q3_Newspaper - Q1_Newspaper
lower_bound_Newspaper = Q1_Newspaper - 1.5 * IQR_Newspaper
upper_bound_Newspaper = Q3_Newspaper + 1.5 * IQR_Newspaper
Sales_Data_no_outliers = Sales_Data[(Sales_Data['Newspaper'] >= lower_bound_Newspaper) & (Sales_Data['Newspaper'] <= upper_bound_Newspaper)]
```

### Feature Scaling

**Standardization**
- Features are standardized using `StandardScaler` to ensure uniform scaling before model training.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Sales_Data_no_outliers.drop(columns=['Sales']))
```

### Train-Test Split

The dataset is split into training and testing sets to evaluate model performance:

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Sales_Data_no_outliers['Sales'], test_size=0.2, random_state=42)
```

## Model Training and Evaluation

### Model Training

A linear regression model is trained using the scaled training data:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Performance Metrics

Evaluate the model using Mean Squared Error (MSE) for both training and testing sets:

```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("Training Mean Squared Error:", mean_squared_error(y_train, y_train_pred))
print("Testing Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
```

### Visualizations

Scatter plots with regression lines for TV, Radio, and Newspaper advertising costs against sales:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Scatter plots and regression lines for TV, Radio, Newspaper vs Sales
plt.tight_layout()
plt.show()
```

## Results

- **Training Mean Squared Error:** [Training MSE]
- **Testing Mean Squared Error:** [Testing MSE]
- **Model Coefficients and Intercept:** Displayed to understand the impact of each feature on sales.

## Conclusion

This project demonstrates the data preparation, training, and evaluation of a linear regression model for predicting sales based on advertising costs. The visualizations and metrics provide insights into how different advertising channels influence sales and the model's accuracy in making predictions.
