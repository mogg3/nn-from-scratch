import numpy as np

def train_test_valid_split(data, train_size, valid_size, random_state=0):
    np.random.seed(random_state)

    data = np.array(data)
    np.random.shuffle(data)
    n = data.shape[1]

    split_one = int(len(data) * train_size)
    split_two = split_one + int(len(data) * valid_size)

    train_data = data[:split_one].T
    y_train = train_data[0]
    X_train = train_data[1:n] / 255.

    valid_data = data[split_one:split_two].T
    y_valid = valid_data[0]
    X_valid = valid_data[1:n] / 255.

    test_data = data[split_two:].T
    y_test = test_data[0]
    X_test = test_data[1:n] / 255.

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def get_all_predictions(net, X_test, y_test):
    all_predictions = []
    for i in range(len(X_test)):
        x = X_test[:, i, None]
        prediction = net.predict(x)
        prediction = np.argmax(prediction, 0)[0]
        label = y_test[i]

        all_predictions.append((prediction,label, x))
    return all_predictions