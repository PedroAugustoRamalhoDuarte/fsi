from pandas import read_csv
from matplotlib import pyplot


def read_data():
    return read_csv('./data/SA_heart.csv')


def clean_data(dataset):
    dataset['famhist'].replace({"Present": 1, "Absent": 0}, inplace=True)


if __name__ == '__main__':
    dataset = read_data()
    clean_data(dataset)
    # 1. Mean and std from every variable

    # Drop not variables columns
    describe_data_set = dataset.drop('id', axis=1)
    describe_data_set = describe_data_set.drop('chd', axis=1)

    # Calculate mean and std
    describe = describe_data_set.describe()
    variables_mean = describe.T['mean']
    variables_std = describe.T['std']

    # Plot mean and std
    variables_mean.plot(kind='bar', title='Média das variáveis')
    pyplot.show()
    variables_std.plot(kind='bar', title='Desvio padrão das variáveis')
    pyplot.show()
