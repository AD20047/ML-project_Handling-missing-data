import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('pima-indians-diabetes.csv')
print('Number of missing entries in each column:')
print(dataset.isnull().sum())

X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)
X = imp_mean.transform(X)
print('\nUpdated matrix after handling missing data:')
print(X)