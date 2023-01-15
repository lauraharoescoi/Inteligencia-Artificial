import random
from typing import Union, List
import treepredict
import clusters
import sys
import matplotlib.pyplot as plt

header=[]
best_thresholds=[0.27,0.29,0.3,0.31,0.32,0.33]

def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(tree: treepredict.DecisionNode, dataset):
    hits = 0
    for row in dataset:
        if treepredict.classify(tree, row) == row[-1]:
            hits +=1
    return hits/len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)

def divide_folds(dataset, k, seed):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    assigned = 0
    n_rows = len(dataset)
    folds_index = []
    choices = list(range(n_rows))
    random.shuffle(choices)
    for fold in range(k):
        folds_index.append([i for i in choices[fold::k]])
    folds = []
    for fold in range(k):
        folds.append([row for (i, row) in enumerate(dataset) if i in folds_index[fold]])

    return folds


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    folds = divide_folds(dataset, k, seed)
    acc = []
    best_accuracy = 0
    for index in range(k):
        test= folds[index]
        train=[]
        for i in range(k):
            if i != index:
                train += (folds[i])
        tree = treepredict.buildtree(train, treepredict.gini_impurity, beta)
        tree = treepredict.prune(tree, threshold)
        current_acc = get_accuracy(tree, test)
        acc.append(current_acc)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_train = train
            best_test = test
    accuracy_mean = agg(acc)
    return (accuracy_mean, best_accuracy, best_train, best_test)


def test_threshold(beta):
    print("\n------- Pruning --------\n")
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "iris.csv"

    header, data = treepredict.read(filename)
    accuracy, top_accuracy, train, test = cross_validation(data, 4, mean, 1, treepredict.gini_impurity, 0, 0)
    tree = treepredict.buildtree(train, treepredict.gini_impurity, beta)
    for threshold in best_thresholds:
        test_tree = treepredict.prune(tree, threshold)
        current_acc = get_accuracy(test_tree, test)
        print(f"{threshold} : {current_acc}")

test_threshold(0)

def values_of_k():
    row_names, headers, data = clusters.readfile("blogdata_full.txt")
    print("\n------- Elbow method --------\n")
    k=input("Max number of k: ")
    k_values = []
    sum_distances = []
    for i in range(1, int(k)+1):
        _, dist = clusters.kcluster(data, distance=clusters.euclidean_squared, k=i, execution=30)
        k_values.append(i)
        sum_distances.append(dist)
        print(k_values[i-1], sum_distances[i-1])

    plt.title("Elbow method")
    plt.plot(k_values, sum_distances)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Sum of distances")
    plt.show()

values_of_k()
