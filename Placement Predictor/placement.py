# Importing all the libraries

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
"""# Importing the dataset"""

df = pd.read_csv("Placement_data.csv")

"""# Making dataset suitable by removing unwanted variables"""

df = df.drop(["ssc_b", "hsc_b", "status", "hsc_s", "gender"], axis=1)
df = df[["degree_t", "workex", "specialisation", "ssc_p","hsc_p", "degree_p", "etest_p", "mba_p", "salary"]]
df['salary'] = df['salary'].replace(np.nan, 0)
df['salary'] = (df['salary'] > 0)

"""# Making dummy variables"""

df['workex'] = df['workex'].astype('category')
df['degree_t'] = df['degree_t'].astype('category')
df = pd.get_dummies(df)
df = df.drop(["workex_No", "degree_t_Others"], axis=1)
df = df[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p','workex_Yes', 'salary']]
df.ssc_p.value_counts().sort_index()

"""# Splitting the model"""
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# Making a test model to find out important parameters

model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# Finding the best parameters

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


print(rf_random.best_params_)


"""# Feature Scaling"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""# Fitting Classifier to training set"""
from sklearn.model_selection import cross_val_score
from statistics import mean
print(mean(cross_val_score(RandomForestClassifier(), X, y, cv=10)))

"""# Predicting the Test Set results"""
# define the model & fit the model
classifier = RandomForestClassifier(n_estimators=400, min_samples_split=2,min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy test to find out whether the model is overfitting or not
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, classifier.predict(X_train))
test_acc = accuracy_score(y_test, classifier.predict(X_test))
print(train_acc)
print(test_acc)
