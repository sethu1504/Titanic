import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import csv
import math

# Data Loading
data = pd.read_csv('../datasets/train.csv')
submit_data = pd.read_csv('../datasets/test.csv')
passenger_ids = submit_data['PassengerId']
del data['PassengerId']
data_y = pd.DataFrame(data, columns=['Survived'])
del data['Survived']
del data['Ticket']
del submit_data['Ticket']
data_names = pd.DataFrame(data, columns=['Name'])
submit_data_names = pd.DataFrame(submit_data, columns=['Name'])
del submit_data['PassengerId']

# FEATURE ENGINEERING - Data Preprocessing

# Introduce New variables
data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
submit_data['Title'] = submit_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
del data['Name']
del submit_data['Name']

# Numeric Features
numeric_features = list(data.dtypes[data.dtypes != 'object'].index)
numeric_features.remove('Pclass')
age_dict = {}
age_dict_submit = {}
for feature in numeric_features:
    if feature == 'Age':
        # Replace NA age by computing the mean of ages based on Sex, Pclass and Sex
        for index, row in data.iterrows():
            age = row['Age']
            if math.isnan(age):
                title = row['Title']
                sex = row['Sex']
                pclass = row['Pclass']
                key = str(pclass) + str(title) + str(sex)
                mean_age = age_dict.get(key)
                if mean_age is None:
                    mean_age = np.mean(data[(data['Title'] == title) & (data['Sex'] == sex) & (data['Pclass'] == pclass)]['Age'])
                    if math.isnan(mean_age):
                        mean_age = np.mean(data[(data['Sex'] == sex) & (data['Pclass'] == pclass)]['Age'])
                    age_dict[key] = mean_age
                data.at[index, 'Age'] = mean_age
        for index, row in submit_data.iterrows():
            age = row['Age']
            if math.isnan(age):
                title = row['Title']
                sex = row['Sex']
                pclass = row['Pclass']
                key = str(pclass) + str(title) + str(sex)
                mean_age = age_dict_submit.get(key)
                if mean_age is None:
                    mean_age = np.mean(submit_data[(submit_data['Title'] == title) & (submit_data['Sex'] == sex) & (submit_data['Pclass'] == pclass)]['Age'])
                    if math.isnan(mean_age):
                        mean_age = np.mean(submit_data[(submit_data['Sex'] == sex) & (submit_data['Pclass'] == pclass)]['Age'])
                    age_dict_submit[key] = mean_age
                submit_data.at[index, 'Age'] = mean_age
    else:
        data[feature].fillna(np.mean(data[feature]), inplace=True)
        submit_data[feature].fillna(np.mean(submit_data[feature]), inplace=True)

# Classification Variables
classi_features = list(data.dtypes[data.dtypes == 'object'].index)
classi_features.append('Pclass')
for feature in classi_features:
    feature_values = data[feature].unique()
    for elem in feature_values:
        data[str(feature + '_' + str(elem))] = pd.Series(data[feature] == elem, dtype=int)
        submit_data[str(feature + '_' + str(elem))] = pd.Series(submit_data[feature] == elem, dtype=int)
    del data[feature]
    del submit_data[feature]

# Hyper Parameters Tuning
# Ridge Tuning ###
# ridge_model = RidgeClassifier()
# grid_param = {'alpha': [1.0, 0.1, 0.001, 0.0001, 0.00001]}
# grid_model = GridSearchCV(ridge_model, grid_param, verbose=1, cv=10, n_jobs=-1, scoring='accuracy')
# grid_model.fit(train_data_x, train_data_y.values.ravel())
# tuned_alpha = grid_model.best_params_.get('alpha')
# tuned alpha - 0.001

# XGB Tuning
# xgb_model = XGBClassifier()
# grid_param = {'learning_rate': [0.001, 0.0001],
#           'booster': ['gblinear', 'gbtree', 'dart'],
#           'n_estimators': [200, 300],
#           'reg_alpha': [0.1, 0.01, 0.001],
#           'reg_lambda': [0.1, 0.01, 0.001]}
# grid_model = GridSearchCV(xgb_model, grid_param, verbose=1, cv=10, n_jobs=-1, scoring='accuracy')
# grid_model.fit(train_data_x, train_data_y.values.ravel())
# {'n_estimators': 200, 'reg_lambda': 0.001, 'learning_rate': 0.0001, 'reg_alpha': 0.01, 'booster': 'gbtree'}

# Random Forest Tuning
# rnd_for_model = RandomForestClassifier()
# grid_param = { "n_estimators": [100, 200, 500],
#                "criterion": ["gini", "entropy"],
#                "max_features": ['sqrt','log2', 0.2, 0.5, 0.8],
#                "max_depth": [3, 4, 6, 10],
#                "min_samples_split": [2, 5, 20, 50]}
# grid_model = GridSearchCV(rnd_for_model, grid_param, verbose=1, cv=10, n_jobs=-1, scoring='accuracy')
# grid_model.fit(train_data_x, train_data_y.values.ravel())
# # {'max_features': 0.2, 'min_samples_split': 2, 'criterion': 'gini', 'max_depth': 10, 'n_estimators': 500}
# 10 fold cv - {'max_features': 0.8, 'min_samples_split': 5, 'criterion': 'entropy', \
# 'max_depth': 10, 'n_estimators': 500}
# print grid_model.best_score_
# print grid_model.best_params_

# Training Model
ridge_model = RidgeClassifier(alpha=0.001, normalize=True)
xgb_model = XGBClassifier(n_estimators=200, reg_lambda=0.001, learning_rate=0.0001, reg_alpha=0.01, booster='gbtree')
rnd_forest_model = RandomForestClassifier(max_features=0.2, min_samples_split=2, criterion='gini', max_depth=10,
                                          n_estimators=500)
rnd_forest_model.fit(data, data_y.values.ravel())
xgb_model.fit(data, data_y.values.ravel())
ridge_model.fit(data, data_y.values.ravel())

# Ensemble
ens_model = VotingClassifier(estimators=[('RF', rnd_forest_model), ('RR', ridge_model), ('XGB', xgb_model)],
                             weights=[1, 1, 1])
ens_model.fit(data, data_y.values.ravel())

# Performance Analysis
print 'Random Forest:'
print 'Cross Val Score = ' + str(np.mean(cross_val_score(rnd_forest_model, data, data_y.values.ravel(),
                                                         scoring='accuracy', cv=10)))

print 'XGB Classifier:'
print 'Cross Val Score = ' + str(np.mean(cross_val_score(xgb_model, data, data_y.values.ravel(),
                                                         scoring='accuracy', cv=10)))

print 'Ridge Classifier:'
print 'Cross Val Score = ' + str(np.mean(cross_val_score(ridge_model, data, data_y.values.ravel(),
                                                         scoring='accuracy', cv=10)))

print 'Ensemble Model:'
print 'Cross Val Score = ' + str(np.mean(cross_val_score(ens_model, data, data_y.values.ravel(),
                                                         scoring='accuracy', cv=10)))

# Predictions
rnd_predictions = rnd_forest_model.predict(submit_data)
xgb_predictions = xgb_model.predict(submit_data)
ridge_predictions = ridge_model.predict(submit_data)
ens_predictions = ens_model.predict(submit_data)

# Submission
out_rnd = csv.writer(open('Submission_Random.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out_xgb = csv.writer(open('Submission_XGB.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out_ridge = csv.writer(open('Submission_Ridge.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out_ens = csv.writer(open('Submission_Ensemble.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out_rnd.writerow(['PassengerId', 'Survived'])
out_xgb.writerow(['PassengerId', 'Survived'])
out_ens.writerow(['PassengerId', 'Survived'])
out_ridge.writerow(['PassengerId', 'Survived'])
for i in range(len(rnd_predictions)):
    passId = passenger_ids[i]
    out_rnd.writerow([passId, rnd_predictions[i]])
    out_xgb.writerow([passId, xgb_predictions[i]])
    out_ens.writerow([passId, ens_predictions[i]])
    out_ridge.writerow([passId, ridge_predictions[i]])

