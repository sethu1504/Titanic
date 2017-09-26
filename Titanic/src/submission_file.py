import pandas as pd
import math
import csv


##1, Age, Sex, Fare
coeffs = [0.6833409214114718, -0.002790749170381012, -0.5327407094171379, 0.315478679390216, 0.16347092438410696, 0.08627897057435108, 0.019018404094418462, 0.0001069822343541833, -0.03855272193501939, 0.0001069822343541833]

df = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/test.csv')

x = []
passId = []
for index, row in df.iterrows():
    sex = 1 if row['Sex'] == 'male' else 0
    age = 0 if math.isnan(row['Age']) else row['Age']
    fare = 0 if math.isnan(row['Fare']) else row['Fare']
    embark = row['Embarked']
    sibsp = row['SibSp']
    parch = row['Parch']

    pclass_1 = 0
    pclass_2 = 0
    pclass = row['Pclass']
    if pclass == 1:
        pclass_1 = 1
    elif pclass == 2:
        pclass_2 = 1

    embark_s = 0
    embark_c = 0
    if embark == 'S':
        embark_s = 1
    elif embark == 'C':
        embark_c = 1
    passenger_id = row['PassengerId']
    x.append([1, age, sex, pclass_1, pclass_2, embark_c, embark_s, fare, sibsp, parch])
    passId.append(passenger_id)
out = csv.writer(open('Submission.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
out.writerow(['PassengerId', 'Survived'])

for i in range(len(passId)):
    passenger_id = passId[i]
    x_each = x[i]
    print_value = 1
    output = 0
    for j in range(len(x_each)):
        output += x_each[j] * coeffs[j]
    if output < 0.7 or output > 1.5:
        print_value = 0
    out.writerow([passenger_id, print_value])
