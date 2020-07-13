# Importing all the libraries
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

"""# Importing the dataset"""

df = pd.read_csv("Placement_data.csv")

"""# Making dataset suitable by removing unwanted variables"""

df = df.drop(["ssc_b","hsc_b","status","hsc_s","gender"],axis=1)
df = df[["degree_t","workex","specialisation","ssc_p","hsc_p","degree_p","etest_p","mba_p","salary"]]
df['salary'] = df['salary'].replace(np.nan,0)
df['salary']=(df['salary']>0)

"""# Making dummy variables"""

df['workex'] = df['workex'].astype('category')
df['degree_t'] = df['degree_t'].astype('category')
df = pd.get_dummies(df)
df = df.drop(["workex_No","degree_t_Others"],axis = 1)
df = df[['ssc_p','hsc_p','degree_p','etest_p','mba_p','degree_t_Comm&Mgmt','degree_t_Sci&Tech',
'workex_Yes', 'specialisation_Mkt&Fin','specialisation_Mkt&HR','salary']]

"""# Splitting the model"""

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""# Feature Scaling"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""# Fitting Classifier to training set"""

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

"""# Predicting the Test Set results"""

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)