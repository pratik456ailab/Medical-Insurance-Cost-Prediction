Medical Insurance Cost Prediction Using Machine Learning
Predicting healthcare costs is a critical task for both insurers and policyholders. In this , I walk through building a machine learning model to predict medical insurance charges based on key personal attributes. The project utilizes Python, Pandas, and Scikit-learn to perform data analysis and model training.
________________________________________
Dataset Overview
We used the publicly available Medical Cost Personal Datasets containing demographic data of individuals, including:
•	Age: Age of primary beneficiary
•	Sex: Gender (male/female)
•	BMI: Body mass index
•	Children: Number of children covered by insurance
•	Smoker: Smoking status
•	Region: Residential area
•	Charges: Individual medical costs billed by health insurance (target variable)
________________________________________
Step 1: Importing Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
________________________________________
Step 2: Loading the Data
df = pd.read_csv('insurance.csv')
df.head()
________________________________________
Step 3: Exploratory Data Analysis (EDA)
We first explored the structure and basic statistics of the dataset.
df.info()
df.describe()
df.isnull().sum()
Data Visualizations
sns.distplot(df['age'])
sns.countplot(x='sex', data=df)
sns.boxplot(x='smoker', y='charges', data=df)
From the boxplot, it's clear that smokers incur significantly higher medical charges.
________________________________________
Step 4: Data Preprocessing
We encoded categorical variables using pd.get_dummies():
df_encoded = pd.get_dummies(df, drop_first=True)
Then we split the features and labels:
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
________________________________________
Step 5: Model Training
We used a simple linear regression model for prediction.
model = LinearRegression()
model.fit(X_train, y_train)
________________________________________
Step 6: Model Evaluation
We evaluated the model using MAE, MSE, and R-squared metrics:
y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}, R^2 Score: {r2}")
________________________________________
Conclusion
This project demonstrated how simple linear regression can provide meaningful predictions in real-world cost estimation scenarios. While this model offers a baseline, future improvements could involve:
•	Trying advanced models (e.g., Random Forest, XGBoost)
•	Hyperparameter tuning
•	Incorporating more detailed health records
________________________________________
Key Learnings
•	Data preprocessing is crucial for ML model performance.
•	Categorical encoding enables the model to process non-numerical inputs.
•	Linear regression, while simple, offers valuable insights and a solid baseline.
