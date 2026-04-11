import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset('iris')

print(data.head())
print(data.shape)
print(data.columns)


sns.scatterplot(x='sepal_length', y='petal_length', data=data)
plt.title('Sepal Length vs Petal Length')
plt.show()

sns.boxplot(x='species', y='sepal_length', data=data)
plt.title('Sepal Length Distribution by Species')
plt.show()

sns.countplot(x='species', hue='petal_length', data=data)
plt.title('Petal Length Count by Species')
plt.show()

