import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


layers = 1


def format_data(X, y, batches_size=1000):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2
    )

    X_train = X_train / 255
    X_test = X_test / 255
    new_y_train = []
    for y in y_train:
        new_y_train.append(int(y))
    new_y_test = []
    for y in y_test:
        new_y_test.append(int(y))
    y_train = np.array(new_y_train)
    y_test = np.array(new_y_test)

    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T

    X_train_array = np.hsplit(X_train, batches_size)
    y_train_array = np.hsplit(y_train, batches_size)

    return X_train_array, X_test, y_train_array, y_test


def initialize_params(hidden_layer=10):
    weights1 = np.random.rand(hidden_layer, 784) - 0.5
    bias1 = np.random.rand(hidden_layer, 1) - 0.5
    if layers == 2:
        weights2 = np.random.rand(hidden_layer, hidden_layer) - 0.5
        bias2 = np.random.rand(hidden_layer, 1) - 0.5
        weights3 = np.random.rand(10, hidden_layer) - 0.5
        bias3 = np.random.rand(10, 1) - 0.5
    else:
        weights2 = np.random.rand(10, hidden_layer) - 0.5
        bias2 = np.random.rand(10, 1) - 0.5
        weights3 = None
        bias3 = None
    return weights1, bias1, weights2, bias2, weights3, bias3


def ReLU(Z):
    return np.maximum(Z, 0)


def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(weights1, bias1, weights2, bias2, X, weights3=None, bias3=None):
    Z1 = weights1.dot(X) + bias1
    A1 = ReLU(Z1)
    Z2 = weights2.dot(A1) + bias2
    if layers == 2:
        A2 = ReLU(Z2)
        Z3 = weights3.dot(A2) + bias3
        A3 = softmax(Z3)
    else:
        A2 = softmax(Z2)
        Z3 = None
        A3 = None
    return Z1, A1, Z2, A2, Z3, A3


def ReLU_deriv(Z):
    return Z > 0


def sigmoid_deriv(Z):
    d = np.exp(Z) / ((np.exp(Z) + 1) ** 2)
    return d


def get_output_array(Y):
    output_array = np.zeros((Y.size, 10))
    output_array[np.arange(Y.size), Y] = 1
    output_array = output_array.T
    return output_array


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, Z3=None, A3=None, W3=None):
    output_array = get_output_array(Y)
    m = Y.size
    if layers == 2:
        dZ3 = A3 - output_array
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3)
        dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    else:
        dW3 = None
        db3 = None
        dZ2 = A2 - output_array
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(
    W1,
    b1,
    W2,
    b2,
    dW1,
    db1,
    dW2,
    db2,
    learn_power,
    W3=None,
    b3=None,
    dW3=None,
    db3=None,
):
    W1 = W1 - learn_power * dW1
    b1 = b1 - learn_power * db1
    W2 = W2 - learn_power * dW2
    b2 = b2 - learn_power * db2
    if layers == 2:
        W3 = W3 - learn_power * dW3
        b3 = b3 - learn_power * db3
    else:
        W3 = None
        b3 = None
    return W1, b1, W2, b2, W3, b3


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def make_predictions(X, W1, b1, W2, b2, W3=None, b3=None):
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, X, W3, b3)
    if layers == 2:
        predictions = get_predictions(A3)
    else:
        predictions = get_predictions(A2)
    return predictions


def train(X_array, Y_array, learn_power, iterations):
    W1, b1, W2, b2, W3, b3 = initialize_params()
    for i in range(iterations):
        for X, Y in zip(X_array, Y_array):
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, X, W3, b3)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(
                Z1, A1, Z2, A2, W1, W2, X, Y, Z3, A3, W3
            )
            W1, b1, W2, b2, W3, b3 = update_params(
                W1, b1, W2, b2, dW1, db1, dW2, db2, learn_power, W3, b3, dW3, db3
            )
        if i % 10 == 0:
            print("Iteration: ", i)
            if layers == 2:
                predictions = get_predictions(A3)
            else:
                predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3


if __name__ == "__main__":
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X_train_array, X_test, y_train_array, y_test = format_data(X, y, 1000)
    weights1, bias1, weights2, bias2, weights3, bias3 = train(
        X_train_array, y_train_array, 0.1, 50
    )
    dev_predictions = make_predictions(
        X_test, weights1, bias1, weights2, bias2, weights3, bias3
    )
    print("\n on test data")
    print(get_accuracy(dev_predictions, y_test))
    image_list = []
    number_list = []
    for i in range(len(dev_predictions)):
        if dev_predictions[i] != y_test[i]:
            image = X_test[:, i]
            image = image.reshape((28, 28)) * 255
            image_list.append(image)
            number_list.append((dev_predictions[i], y_test[i]))
    for i in range(min(10, len(image_list))):
        plt.imshow(image_list[i], cmap=plt.get_cmap("gray"))
        plt.savefig(f"nr_{i}_guess_{number_list[i][0]}_written_{number_list[i][1]}.png")
