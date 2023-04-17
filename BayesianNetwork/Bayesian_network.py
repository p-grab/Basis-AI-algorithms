import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

col_names = [
    "class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
    "index",
]


def calculate_p_y(data, Y):
    classes = set(data[Y])
    p_y = []
    for i in classes:
        p_y.append(len(data[data[Y] == i]) / len(data))
    return p_y


def calculate_p_x_given_y(data, feat_name, feat_val, Y, label):
    data_given_y = data[data[Y] == label]

    mean, std = data_given_y[feat_name].mean(), data_given_y[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(
        -((feat_val - mean) ** 2 / (2 * std**2))
    )
    return p_x_given_y


def naive_bayes_alg(data, X, Y, feat_names):
    feature_names = list(data.columns)[1:]
    p_y = calculate_p_y(data, Y)
    labels = sorted(list(set(data[Y])))
    Y_pred = []

    for row in X:
        p_x_given_y = [1] * len(labels)
        for j in range(len(labels)):
            for i in feat_names:
                i = col_names.index(i) - 1
                p_x_given_y[j] *= calculate_p_x_given_y(
                    data, feature_names[i], row[i], Y, labels[j]
                )
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = p_x_given_y[j] * p_y[j]
        Y_pred.append(np.argmax(post_prob) + 1)
    print(i)
    return np.array(Y_pred)


if __name__ == "__main__":
    data = pd.read_csv("BayesianNetwork/wine.data")

    if (data.columns.size) < 15:
        data["index"] = range(1, len(data) + 1)
        data.to_csv("BayesianNetwork/wine.data", index=False, header=None)

    data = pd.read_csv(
        "BayesianNetwork/wine.data", header=None, names=col_names, index_col="index"
    )
    train, test = train_test_split(data, test_size=0.4, shuffle=True)

    X_test = test.iloc[:, 1:].values
    Y_test = test.iloc[:, 0].values

    feat_names = [
        "Alcohol",
        # "Malic acid",
        # "Ash",
        # "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        # "Flavanoids",
        # "Nonflavanoid phenols",
        "Proanthocyanins",
        # "Color intensity",
        # # "Hue",
        # "OD280/OD315 of diluted wines",
        # "Proline",
    ]
    Y_pred = naive_bayes_alg(train, X=X_test, Y="class", feat_names=feat_names)

    print(Y_pred)
    print(Y_test)

    print(confusion_matrix(Y_test, Y_pred))

    from sklearn.metrics import accuracy_score

    print(round(accuracy_score(Y_test, Y_pred), 3))

    from sklearn.metrics import classification_report

    print(classification_report(Y_test, Y_pred))

    # from matplotlib import pyplot as plt

    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.plot(X_test[:, 0], X_test[:, 1])
    # plt.show()
