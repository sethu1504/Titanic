import numpy as np
from sklearn import linear_model
import pandas as pd
import csv

train_data = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/train.csv')

train_data_y = pd.DataFrame(train_data, columns=['Survived'])
train_data_x = pd.DataFrame(train_data, columns=['Age', 'Sex', 'SibSp', 'Parch'])
train_data_x['Age'].fillna(np.mean(train_data_x['Age']), inplace=True)
train_data_x['Sex'] = train_data_x['Sex'].map({'male': 1, 'female': 0})
train_data_x['SibSp'].fillna(np.mean(train_data_x['SibSp']), inplace=True)
train_data_x['Parch'].fillna(np.mean(train_data_x['Parch']), inplace=True)

test_data = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/test.csv')
test_data_y = pd.DataFrame(test_data, columns=['Survived'])
test_data_x = pd.DataFrame(test_data, columns=['Age', 'Sex', 'SibSp', 'Parch'])
test_data_x['SibSp'].fillna(np.mean(test_data_x['SibSp']), inplace=True)
test_data_x['Parch'].fillna(np.mean(test_data_x['Parch']), inplace=True)
test_data_x['Age'].fillna(np.mean(test_data_x['Age']), inplace=True)
test_data_x['Sex'] = test_data_x['Sex'].map({'male': 1, 'female': 0})

passengerIds = test_data['PassengerId']


model = linear_model.LinearRegression()
model.fit(train_data_x.values, train_data_y.values)
predicted_y = model.predict(test_data_x.values)

out = csv.writer(open('Submission.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out.writerow(['PassengerId', 'Survived'])

for i in range(len(passengerIds)):
    print_value = 0
    passId = passengerIds[i]
    y = predicted_y[i][0]
    if y >= 0.7:
        print_value = 1
    out.writerow([passId, print_value])