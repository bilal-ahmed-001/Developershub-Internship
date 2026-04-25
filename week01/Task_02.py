import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('dataset.csv')

data.columns = data.columns.str.strip()

print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())

num_cols = [
    'no_of_dependents', 'income_annum', 'loan_term',
    'loan_amount', 'cibil_score', 'residential_assets_value',
    'luxury_assets_value', 'commercial_assets_value',
    'bank_asset_value'
]

cat_cols = ['education', 'self_employed', 'loan_status']

for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("After handling missing values:")
print(data.isnull().sum())

sns.histplot(data['loan_amount'], bins=30, kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x='education', hue='loan_status', data=data)
plt.title("Education vs Loan Status")
plt.show()

sns.boxplot(x='loan_status', y='income_annum', data=data)
plt.title("Income vs Loan Status")
plt.show()

le = LabelEncoder()

for col in cat_cols:
    data[col] = le.fit_transform(data[col])

data = data.drop('loan_id', axis=1)

X = data.drop('loan_status', axis=1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))