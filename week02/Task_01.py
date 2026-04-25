import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, classification_report
import shap
df = pd.read_csv('bank_marketing.csv')
df.head()
df.shape
df['deposit'].value_counts()
df.describe()
df['age'].plot(kind='hist');
df['balance'].plot(kind='hist');
binary_cols = ['default', 'housing', 'loan', 'deposit']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    education_map = {
    'primary': 1,
    'secondary': 2,
    'tertiary': 3,
    'unknown': 0  }
df['education'] = df['education'].map(education_map)

nominal_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

df.shape


X = df.drop('deposit', axis=1)
y = df['deposit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression()
rf_model = RandomForestClassifier(random_state=42)

lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

print("Models trained successfully.")

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]


print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)

shap.initjs() 

for i in range(5):
    print(f"Explanation for Sample {i}:")
    display(shap.force_plot(explainer.expected_value[1], shap_values[1][i], X_test.iloc[i], feature_names=X.columns))

