import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('titanic.csv')

print(data.head())
print(data.shape)
print(data.columns)

sns.scatterplot(x='Age', y='Fare', data=data)
plt.title('Age vs Fare')
plt.show()

sns.boxplot(x='Survived', y='Age', data=data)
plt.title('Age Distribution by Survival')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Count by Passenger Class')
plt.show()

