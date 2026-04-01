import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv('insurance.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Convert categorical variables to numeric
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)

print("Columns after encoding:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

plt.figure(figsize=(18, 12))

# 1. Age vs Charges
plt.subplot(2, 3, 1)
sns.scatterplot(x='age', y='charges', data=df, alpha=0.6)
sns.regplot(x='age', y='charges', data=df, scatter=False, color='red')
plt.title('Age vs Insurance Charges')
plt.xlabel('Age')
plt.ylabel('Charges ($)')

# 2. BMI vs Charges
plt.subplot(2, 3, 2)
sns.scatterplot(x='bmi', y='charges', data=df, alpha=0.6)
sns.regplot(x='bmi', y='charges', data=df, scatter=False, color='red')
plt.title('BMI vs Insurance Charges')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')

# 3. Smoker vs Charges
plt.subplot(2, 3, 3)
sns.boxplot(x='smoker_yes', y='charges', data=df)
plt.title('Smoking Status vs Insurance Charges')
plt.xlabel('Smoker (1 = Yes, 0 = No)')
plt.ylabel('Charges ($)')

# 4. Age & BMI combined with Smoker
plt.subplot(2, 3, 4)
sns.scatterplot(x='age', y='charges', hue='smoker_yes', data=df, alpha=0.7)
plt.title('Age vs Charges by Smoking Status')
plt.xlabel('Age')
plt.ylabel('Charges ($)')

# 5. BMI & Charges by Smoker
plt.subplot(2, 3, 5)
sns.scatterplot(x='bmi', y='charges', hue='smoker_yes', data=df, alpha=0.7)
plt.title('BMI vs Charges by Smoking Status')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')

# 6. Correlation Heatmap
plt.subplot(2, 3, 6)
corr = df.corr()
sns.heatmap(corr[['charges']].sort_values(by='charges', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation with Charges')

plt.tight_layout()
plt.show()

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Model Coefficients (impact on charges):")
print(coefficients)


# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE)    : ${mae:,.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
print(f"R² Score                     : {model.score(X_test, y_test):.4f}")