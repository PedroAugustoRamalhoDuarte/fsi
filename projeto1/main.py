import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn import svm, datasets
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, KFold


def read_data():
    return read_csv('./data/SA_heart.csv')


def clean_data(dataset):
    dataset['famhist'].replace({"Present": 1, "Absent": 0}, inplace=True)


def data_fold(dataset, fold_number):
    FOLD_PERCENT = 0.1
    rows = dataset.shape[0]
    test_rows = int(rows * FOLD_PERCENT)
    test_row_initial_index = test_rows * fold_number
    test_data = dataset.iloc[test_row_initial_index:test_row_initial_index + test_rows]
    train_data = dataset.drop(labels=range(test_row_initial_index, test_row_initial_index + test_rows), axis=0)
    return train_data, test_data


def split_variables_labels(dataset):
    return dataset.drop('chd', axis=1), dataset.loc[:, 'chd']


if __name__ == '__main__':
    dataset = read_data()
    clean_data(dataset)
    # 1. Mean and std from every variable

    # Drop not variables columns
    variables_dataset_with_label = dataset.drop('id', axis=1)
    variables_dataset = variables_dataset_with_label.drop('chd', axis=1)

    # Calculate mean and std
    describe = variables_dataset.describe()
    variables_mean = describe.T['mean']
    variables_std = describe.T['std']

    # Plot mean and std
    # variables_mean.plot(kind='bar', title='Média das variáveis')
    # pyplot.show()
    # variables_std.plot(kind='bar', title='Desvio padrão das variáveis')
    # pyplot.show()

    # 2. Prediction with CART
    # y = variables_dataset_with_label.loc[:, 'chd']
    # x = variables_dataset_with_label
    # k_fold = StratifiedKFold(n_splits=10, random_state=111, shuffle=True)
    #
    # mats = []
    # for train_index, test_index in k_fold.split(x, y):
    #     mats.append(confusion_matrix(y_train[test_index], y_pred[test_index]))
    # predicted_targets = np.array([])
    # actual_targets = np.array([])
    #
    # for train_ix, test_ix in k_fold.split(x):
    #     print(train_ix)
    #     print(test_ix)
    #     train_x, train_y, test_x, test_y = x[train_ix], y[train_ix], x[test_ix], y[test_ix]
    #
    #     clf = tree.DecisionTreeClassifier().fit(train_x, train_y)
    #
    #     predicted_labels = clf.predict(test_x)
    #
    #     predicted_targets = np.append(predicted_targets, predicted_labels)
    #     actual_targets = np.append(actual_targets, test_y)
    #
    #     print(confusion_matrix(test_y, predicted_labels))

    for i in range(0, 10):
        train_data, test_data = data_fold(dataset, i)
        variables, labels = split_variables_labels(train_data)
        test_data_var, test_data_labels = split_variables_labels(train_data)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(variables, labels)

        predict_labels = clf.predict(test_data_var)
        print(confusion_matrix(test_data_labels, predict_labels))
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        # print()
        # print(test_data_labels)
        # tree.plot_tree(clf)
        # pyplot.show()