# First steps with machine learning: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# 3 Summarize the dataset
def dataset_dimensions(dataset):
    print(f"Shape: {dataset.shape}")
    print(f"Head:\n {dataset.head(5)}")
    print(f"Descriptions:\n {dataset.describe()}")
    print(f"Class distribution:\n {dataset.groupby('class').size()}")


# 4 Data visualization
def data_visualization(dataset):
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()
    # histograms
    dataset.hist()
    pyplot.show()
    # scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()


# 5. Evaluate Some Algorithms
def split_dataset(dataset):
    # Split-out validation dataset
    array = dataset.values
    x = array[:, 0:4]
    y = array[:, 4]
    return train_test_split(x, y, test_size=0.20, random_state=1)


def select_best_model(x_train, y_train, print_comparison=False):
    # Spot Check Algorithms
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('SVM', SVC(gamma='auto'))]

    # evaluate each model in turn
    results = []
    names = []
    best_model = {
        "mean": 0,
        "model": models[0],
        "model_name": 'Other'
    }

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)

        if cv_results.mean() > best_model["mean"]:
            best_model = {
                "mean": cv_results.mean(),
                "model": model,
                "model_name": name
            }

        if print_comparison:
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    if print_comparison:
        # Compare Algorithms
        pyplot.boxplot(results, labels=names)
        pyplot.title('Algorithm Comparison')
        pyplot.show()

    return best_model


if __name__ == '__main__':
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)
    # dataset_dimensions(dataset)
    # data_visualization(dataset)

    x_train, x_validation, y_train, y_validation = split_dataset(dataset)
    best_model = select_best_model(x_train, y_train)
    # Predictions
    # Make predictions on validation dataset
    print(f'Algorithm used: {best_model["model_name"]}')
    model = best_model["model"]
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    # Evaluate predictions
    print(accuracy_score(y_validation, predictions))
    print(confusion_matrix(y_validation, predictions))
    print(classification_report(y_validation, predictions))
