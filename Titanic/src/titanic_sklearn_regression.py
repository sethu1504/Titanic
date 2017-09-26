from sklearn import linear_model
import pandas as pd
import random
import numpy as np


def split_data(df):
    test = []
    train = []
    for index, row in df.iterrows():
        d = dict({'PassengerId': row['PassengerId'], 'Survived': row['Survived'], 'Pclass': row['Pclass'], 'Name': row['Name'],
                             'Sex': row['Sex'], 'Age': row['Age'], 'SibSp': row['SibSp'], 'Parch': row['Parch'],
                             'Ticket': row['Ticket'], 'Fare': row['Fare'], 'Cabin': row['Cabin'], 'Embarked': row['Embarked']})
        if random.random() < 0.75:
            train.append(d)
        else:
            test.append(d)

    return pd.DataFrame(train), pd.DataFrame(test)


###Load and split Data###

data = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/train.csv')
train_data, test_data = split_data(data)

####Data preprocessing###

###Train Data####
train_data_y = pd.DataFrame(train_data, columns=['Survived'])
train_data_x = pd.DataFrame(train_data, columns=['Age', 'Sex', 'SibSp', 'Parch'])
train_data_x['Age'].fillna(np.mean(train_data_x['Age']), inplace=True)
train_data_x['SibSp'].fillna(np.mean(train_data_x['SibSp']), inplace=True)
train_data_x['Parch'].fillna(np.mean(train_data_x['Parch']), inplace=True)
train_data_x['Sex'] = train_data_x['Sex'].map({'male': 1, 'female': 0})

####Test Data###
test_data_y = pd.DataFrame(test_data, columns=['Survived'])
test_data_x = pd.DataFrame(test_data, columns=['Age', 'Sex', 'SibSp', 'Parch'])
test_data_x['SibSp'].fillna(np.mean(test_data_x['SibSp']), inplace=True)
test_data_x['Parch'].fillna(np.mean(test_data_x['Parch']), inplace=True)
test_data_x['Age'].fillna(np.mean(test_data_x['Age']), inplace=True)
test_data_x['Sex'] = test_data_x['Sex'].map({'male': 1, 'female': 0})

###Modelling###

model = linear_model.LinearRegression()
model.fit(train_data_x.values, train_data_y.values)
predicted_y = model.predict(test_data_x.values)
correct = 0
for i in range(len(predicted_y)):
    y = predicted_y[i][0]
    actual_y = test_data_y.values[i][0]
    if y >= 0.7:
        if actual_y == 1:
            correct += 1
    else:
        if actual_y == 0:
            correct += 1
print "accuracy = " + str(correct/float((len(test_data_y.values))))

