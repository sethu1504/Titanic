import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/train.csv')

    x_data = pd.DataFrame(data, columns=['Age', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked'])
    y_data = pd.DataFrame(data, columns=['Survived'])

    x_data['Age'].fillna(np.mean(x_data['Age']), inplace=True)
    x_data['SibSp'].fillna(np.mean(x_data['SibSp']), inplace=True)
    x_data['Parch'].fillna(np.mean(x_data['Parch']), inplace=True)
    x_data.loc[x_data['Sex'] == 'male', 'Sex'] = 1
    x_data.loc[x_data['Sex'] == 'female', 'Sex'] = 0
    x_data['Pclass1'] = pd.Series([0] * len(x_data), index=x_data.index)
    x_data.loc[x_data['Pclass'] == 1, 'Pclass1'] = 1
    x_data['Pclass2'] = pd.Series([0] * len(x_data), index=x_data.index)
    x_data.loc[x_data['Pclass'] == 2, 'Pclass2'] = 1
    del x_data['Pclass']
    x_data['S_Embarked'] = pd.Series([0] * len(x_data), index=x_data.index)
    x_data.loc[x_data['Embarked'] == 'S', 'S_Embarked'] = 1
    x_data['C_Embarked'] = pd.Series([0] * len(x_data), index=x_data.index)
    x_data.loc[x_data['Embarked'] == 2, 'C_Embarked'] = 1
    del x_data['Embarked']

    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_data, y_data)

    reg_model = Ridge(alpha=0.00000000000001, normalize=True)
    reg_model.fit(x_train_data.values, y_train_data.values.ravel())

    predictions = reg_model.predict(x_train_data.values)
    predictions_df = pd.DataFrame()
    predictions_df['Prediction'] = pd.Series(predictions)
    predictions_df['Actual'] = y_train_data.values
    predictions_df['Sex'] = x_train_data['Sex'].values
    predictions_df['Age'] = x_train_data['Age'].values

    print predictions_df[predictions_df['Sex'] == 1][['Prediction', 'Age']].head()

    k_model = KNeighborsClassifier()
    k_model.fit(predictions_df[predictions_df['Sex'] == 1][['Prediction', 'Age']].values, predictions_df[predictions_df['Sex'] == 1]['Actual'].values)

    predictions_df_1 = predictions_df[predictions_df['Actual'] == 1]
    predictions_df_0 = predictions_df[predictions_df['Actual'] == 0]

    # plt.figure(1)
    # plt.hist(predictions_df[predictions_df['Actual'] == 1]['Prediction'].values)
    # plt.title('Survived Prediction Values')
    #
    # plt.figure(2)
    # plt.hist(predictions_df[predictions_df['Actual'] == 0]['Prediction'].values)
    # plt.title('Dead Prediciton values')

    # plt.figure(3)
    # plt.hist(predictions_df_1[predictions_df_1['Sex'] == 1]['Prediction'].values)
    # plt.title('Acutal Survivers vs Prediction - Males')
    #
    # plt.figure(4)
    # plt.hist(predictions_df_1[predictions_df_1['Sex'] == 0]['Prediction'].values)
    # plt.title('Acutal Survivers vs Prediction - Females')
    #
    # plt.figure(5)
    # plt.hist(predictions_df_0[predictions_df_0['Sex'] == 1]['Prediction'].values)
    # plt.title('Acutal Dead vs Prediction - Males')
    #
    # plt.figure(6)
    # plt.hist(predictions_df_0[predictions_df_0['Sex'] == 0]['Prediction'].values)
    # plt.title('Acutal Dead vs Prediction - Females')

    plt.figure(7)
    plt.title('Prediction vs Age')

    plt.scatter(predictions_df_1[predictions_df_1['Sex'] == 1]['Prediction'].values,
                predictions_df_1[predictions_df_1['Sex'] == 1]['Age'].values, color='blue')

    plt.scatter(predictions_df_0[predictions_df_0['Sex'] == 1]['Prediction'].values,
                predictions_df_0[predictions_df_0['Sex'] == 1]['Age'].values, color='red')

    plt.show()

    print 'R Score = ' + str(reg_model.score(x_test_data.values, y_test_data.values.ravel()))

    test_data = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/test.csv')
    true_x = pd.DataFrame(test_data, columns=['Age', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked'])

    passIds = test_data['PassengerId']

    true_x['Age'].fillna(np.mean(true_x['Age']), inplace=True)
    true_x['SibSp'].fillna(np.mean(true_x['SibSp']), inplace=True)
    true_x['Parch'].fillna(np.mean(true_x['Parch']), inplace=True)
    true_x.loc[true_x['Sex'] == 'male', 'Sex'] = 1
    true_x.loc[true_x['Sex'] == 'female', 'Sex'] = 0
    true_x['Pclass1'] = pd.Series([0] * len(true_x), index=true_x.index)
    true_x.loc[true_x['Pclass'] == 1, 'Pclass1'] = 1
    true_x['Pclass2'] = pd.Series([0] * len(true_x), index=true_x.index)
    true_x.loc[true_x['Pclass'] == 2, 'Pclass2'] = 1
    del true_x['Pclass']
    true_x['S_Embarked'] = pd.Series([0] * len(true_x), index=true_x.index)
    true_x.loc[true_x['Embarked'] == 'S', 'S_Embarked'] = 1
    true_x['C_Embarked'] = pd.Series([0] * len(true_x), index=true_x.index)
    true_x.loc[true_x['Embarked'] == 2, 'C_Embarked'] = 1
    del true_x['Embarked']

    predictions = reg_model.predict(true_x.values)

    out = csv.writer(open('Submission.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
    out.writerow(['PassengerId', 'Survived'])

    for i in range(len(passIds)):
        survive_value = 0
        passId = passIds[i]
        sex = true_x['Sex'].values[i]
        age = true_x['Age'].values[i]
        if sex == 0:
            if predictions[i] > 0.6:
                survive_value = 1
        else:
            survive_value = k_model.predict([[predictions[i], age]])[0]
        out.writerow([passId, survive_value])
