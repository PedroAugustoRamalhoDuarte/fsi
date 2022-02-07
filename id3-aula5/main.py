from matplotlib import pyplot
from sklearn import tree

# "Febre", "Dor", "Manchas", "Coceiras"
DATA = [
    ["Alta", "Frequente", "Presente", "Intensa"],
    ["Moderada", "Rara", "Presente", "Inexistente"],
    ["Nenhuma", "Frequente", "Ausente", "Inexistente"],
    ["Alta", "Frequente", "Ausente", "Intensa"],
    ["Moderada", "Permanente", "Presente", "Inexistente"],
    ["Moderada", "Permanente", "Ausente", "Intensa"],
    ["Alta", "Permanente", "Presente", "Inexistente"],
    ["Alta", "Permanente", "Ausente", "Inexistente"],
    ["Moderada", "Frequente", "Ausente", "Moderada"],
    ["Moderada", "Frequente", "Ausente", "Intensa"],
    ["Nenhuma", "Rara", "Presente", "Moderada"],
    ["Alta", "Rara", "Presente", "Moderada"],
]

CONVERT_HASH = {
    "Nenhuma": 0,
    "Moderada": 2,
    "Alta": 3,
    "Inexistente": 0,
    "Rara": 1,
    "Frequente": 3,
    "Intensa": 4,
    "Permanente": 5,
    "Ausente": 0,
    "Presente": 1,

}

labels = [
    "Emergência",
    "Urgência",
    "Mais_Exames",
    "Emergência",
    "Urgência",
    "Urgência",
    "Emergência",
    "Emergência",
    "Mais_Exames",
    "Emergência",
    "Mais_Exames",
    "Urgência",
]

data_format = []
for row in DATA:
    row_tmp = []
    for item in row:
        row_tmp.append(CONVERT_HASH[item])
    data_format.append(row_tmp)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(data_format, labels)
tree.plot_tree(clf, feature_names=["Febre", "Dor", "Manchas", "Coceiras"])
pyplot.show()

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
