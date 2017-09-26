from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
import csv
import matplotlib.pyplot as plt

data = pd.read_csv('../datasets/train.csv')
submit_data = pd.read_csv('../datasets/test.csv')
del data['Id']
submit_data_houseIds = submit_data['Id']
del submit_data['Id']
######  FEATURE ENGINEERING ########

### Numeric Features #####
numeric_features = list(data.dtypes[data.dtypes != 'object'].index)
numeric_features.remove('MSSubClass')  #This variable is a classification feature
for feature in numeric_features:
    data[feature].fillna(np.mean(data[feature]), inplace=True)
    data[feature] = np.log1p(data[feature])
    if feature != 'SalePrice':
        submit_data[feature].fillna(np.mean(submit_data[feature]), inplace=True)
        submit_data[feature] = np.log1p(submit_data[feature])

### Classification features - introduce dummy variables ####
classi_features = list(data.dtypes[data.dtypes == 'object'].index)
classi_features.append('MSSubClass')
for feature in classi_features:
    feature_values = data[feature].unique()
    for elem in feature_values:
        data[str(feature + '_' + str(elem))] = pd.Series(data[feature] == elem, dtype=int)
    del data[feature]
    for elem in feature_values:
        submit_data[str(feature + '_' + str(elem))] = pd.Series(submit_data[feature] == elem, dtype=int)
    del submit_data[feature]

### Data split ###
data_y = pd.DataFrame(data, columns=['SalePrice'])
del data['SalePrice']
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data, data_y)

### Hyper Parameter Tuning - alpha ###
#Lasso Tuning
grid_params = {'alpha': [1, 5, 10, 0.1, 0.01, 0.001, 0.0001]}
lasso_model = Lasso(normalize=True)
grid_model = GridSearchCV(lasso_model, grid_params, scoring='neg_mean_squared_error', cv=10, verbose=1, n_jobs=-1)
grid_model.fit(train_data_x, train_data_y)
tuned_alpha = grid_model.best_params_.get('alpha')
#XGB Tuning
# grid_params = {'learning_rate': [0.001, 0.0001],
#           'booster': ['gblinear', 'gbtree'],
#           'n_estimators': [200, 300],
#           'reg_alpha': [0.1, 0.01, 0.001],
#           'reg_lambda': [0.1, 0.01, 0.001]}
# xgb_model = XGBRegressor()
# grid_model = GridSearchCV(xgb_model, grid_params, scoring='neg_mean_squared_error', cv=10, verbose=1, n_jobs=-1)
# grid_model.fit(train_data_x, train_data_y)
# print grid_model.best_estimator_
# print grid_model.best_params_
# Result - {'n_estimators': 300, 'reg_lambda': 0.001, 'learning_rate': 0.001, 'reg_alpha': 0.01, 'booster': 'gbtree'}

### Training ###
lasso_model = Lasso(alpha=tuned_alpha, normalize=True)
lasso_model.fit(train_data_x, train_data_y)
xgb_model = XGBRegressor(n_estimators=300, reg_lambda=0.001, reg_alpha=0.01, learning_rate=0.1, booster='gbtree')
xgb_model.fit(train_data_x, train_data_y)

### Performance ###
print 'Lasso Mode :'
print 'Cross validation score = ' + str(np.mean(cross_val_score(lasso_model, train_data_x, train_data_y, cv=10)))
print 'R score = ' + str(lasso_model.score(train_data_x, train_data_y))
print 'XGBoost Regressor :'
print 'Cross validation score = ' + str(np.mean(cross_val_score(xgb_model, train_data_x, train_data_y, cv=10)))
print 'R score = ' + str(xgb_model.score(train_data_x, train_data_y))

predictions_lasso = lasso_model.predict(submit_data)
predictions_xgb = xgb_model.predict(submit_data)
predicted_house_price = np.expm1(0.60*predictions_lasso + 0.40*predictions_xgb)

out = csv.writer(open('Submission.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out.writerow(['Id', 'SalePrice'])

for i in range(len(submit_data_houseIds)):
    predicted_value = predicted_house_price[i]
    houseId = submit_data_houseIds[i]
    out.writerow([houseId, predicted_value])
