import pandas as pd
import random
import math


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


def generate_x_and_y(df):
    x = []
    y = []
    for index, row in df.iterrows():
        sex = 1 if row['Sex'] == 'male' else 0
        age = 0 if math.isnan(row['Age']) else row['Age']
        fare = 0 if math.isnan(row['Fare']) else row['Fare']
        survived = row['Survived']
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

        x.append([1, age, sex, pclass_1, pclass_2, embark_c, embark_s, fare, sibsp, fare])
        y.append(survived)
    return x, y


def get_predicted_value(x_each, betas):
    predict_value = 0
    for i in range(len(betas)):
        predict_value += x_each[i] * betas[i]
    return predict_value


def get_error(x_each, y_actual, betas):
    return y_actual - get_predicted_value(x_each, betas)


def get_error_squares(x_each, y_actual, betas):
    val = get_error(x_each, y_actual, betas) ** 2
    return val


def get_squared_error_gradients(x_each, y_actual, betas):
    squared_error_gradients = []
    for x_each_i in x_each:
        squared_error_gradients.append(-2 * x_each_i * get_error(x_each, y_actual, betas))
    return squared_error_gradients


def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


def vector_subtract(u, v):
    return [u_i - v_i for u_i, v_i in zip(u, v)]


def scalar_multiply(scalar, vector):
    return [scalar * v_i for v_i in vector]


def minimal_stochastic_prediction(x, y, initial_betas, inital_learning_rate):
    data = zip(x, y)
    betas = initial_betas
    learning_rate = inital_learning_rate
    min_betas = None
    min_value = float('inf')
    counter = 0
    gen_counter = 0
    while counter < 100:
        if gen_counter > 100000:
            print "Breaking after loop exhaustion"
            break
        gen_counter += 1
        error_value = 0
        for i in range(len(x[0])):
            error_value += get_error_squares(x[i], y[i], betas)
        if error_value < min_value:
            min_value = error_value
            counter = 0
            learning_rate = inital_learning_rate
            min_betas = betas
        else:
            learning_rate *= 0.9
            counter += 1
        for x_each, y_actual in data:
            gradient_each = get_squared_error_gradients(x_each, y_actual, betas)
            betas = vector_subtract(betas, scalar_multiply(learning_rate, gradient_each))
    return min_betas


def predict_betas(x, y, beta_initials):
    predicted_betas = minimal_stochastic_prediction(x, y, beta_initials, 0.00001)
    return predicted_betas


def start_training_and_predict(beta_initials):
    df = pd.read_csv('/Users/sethuramanannamalai/Documents/PyCharm/Titanic/datasets/train.csv')
    train, test = split_data(df)
    x, y = generate_x_and_y(train)
    predicted_betas = predict_betas(x, y, beta_initials)
    print predicted_betas
    x, y = generate_x_and_y(test)
    testing_set = zip(x, y)
    dead_list = []
    survive_list = []
    correct = 0
    for x_each, actual_y in testing_set:
        output = 0
        for i in range(len(x_each)):
            output += x_each[i] * predicted_betas[i]
        if actual_y == 0:
            dead_list.append(output)
            if output < 0.7 or output > 1.5:
                correct += 1
        else:
            survive_list.append(output)
            if output >= 0.7 and output <= 1.5:
                correct += 1

    print "Survive List"
    print sorted(survive_list)
    print "Dead List"
    print sorted(dead_list)

    acc = (correct/float(len(y)))*100
    print "Accuracy = " + str(acc)
    return predicted_betas, acc

optimal_betas = [0 for _ in range(10)]
max_accuracy = 0
for i in range(10):
    b, accuracy = start_training_and_predict(optimal_betas)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        optimal_betas = b

print "Max Accuracy = " + str(max_accuracy)
print optimal_betas
