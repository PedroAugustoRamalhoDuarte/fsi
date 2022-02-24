import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn import tree
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay


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
    x = variables_dataset
    y = variables_dataset_with_label.loc[:, 'chd']
    k_fold = KFold(n_splits=10, shuffle=False)

    confusion_matrixs = []
    fold_index = 0
    pyplot.figure()
    for train_ix, test_ix in k_fold.split(x, y):
        train_x, train_y, test_x, test_y = x.iloc[train_ix], y.iloc[train_ix], x.iloc[test_ix], y.iloc[test_ix]

        clf = tree.DecisionTreeClassifier().fit(train_x, train_y)

        predicted_labels = clf.predict(test_x)

        # Roc curve variables
        fpr, tpr, _ = roc_curve(test_y, predicted_labels, pos_label=1)
        roc_auc = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (fold_index + 1, roc_auc))

        # Confusion matrix
        confusion_matrixs.append(confusion_matrix(test_y, predicted_labels))

        fold_index += 1

    # Roc curves plot
    pyplot.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    pyplot.legend(loc="lower right")
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.show()

    # Confusion matrix plot
    confusion_matrixs_total = np.add.reduce(confusion_matrixs)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrixs_total)
    disp.plot()
    pyplot.show()
