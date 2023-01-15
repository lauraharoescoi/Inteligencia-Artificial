#!/usr/bin/env python3
import sys
import collections
from math import log2
from typing import List, Tuple

# Used for typing
Data = List[List]


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            values = line.strip().split(separator)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
    return header, data


def _parse_value(v: str):
    try:
        if float(v) == int(v):
            return int(v)
        return float(v)
    except ValueError:
        return v


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return dict(collections.Counter(row[-1] for row in part))

    # results = collections.Counter()
    # for row in part:
    #     c = row[-1]
    #     results[c] += 1
    # return dict(results)


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v / total) ** 2
    return imp


def entropy(part: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(part)
    results = unique_counts(part)

    probs = (v / total for v in results.values())
    return -sum(p * log2(p) for p in probs)



def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: str):
    return prototype[column] == value


def divideset(part: Data, column: int, value) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    set1 = []
    set2 = []
    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
    
    for row in part:
        if split_function(row, column, value):
            set1.append(row)
        else:
            set2.append(row)

    return (set1, set2)


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None, gain=1):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """
        self.col = col
        self.value = value
        self.tb = tb
        self.fb = fb
        self.results = results
        self.gain = gain


def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)
    
    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None
    
    for col in range(len(part[0]) - 1):
        # Generate the list of possible different values in the current column
        global_values = list(set([row[col] for row in part]))
        global_values.sort(reverse=True)
        for value in global_values:
            # Divide the dataset into two subdatasets
            (set1, set2) = divideset(part, col, value)
            # Calculate the impurity measure for the current split
            p = float(len(set1)) / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            # Update the best criteria if the current split is better
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Create the subnodes
    if best_gain > beta:
        trueBranch = buildtree(best_sets[0], scoref, beta)
        falseBranch = buildtree(best_sets[1], scoref, beta)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch, gain=best_gain)
    else:
        return DecisionNode(results=unique_counts(part))


def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """
    root = DecisionNode()
    queue = [(root, part)]

    while len(queue) > 0:
        current_node, current_part = queue.pop(0)
        if len(current_part) == 0:
            continue
        current_score = scoref(current_part)
        best_gain = 0
        best_criteria = None
        best_sets = None
        for col in range(len(current_part[0])):
            global_values = list(set([row[col] for row in part]))
            global_values.sort(reverse=True)
            for value in global_values:
                (set1, set2) = divideset(current_part, col, value)
                p = float(len(set1)) / len(current_part)
                gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
        if best_gain > beta:
            current_node.col = best_criteria[0]
            current_node.value = best_criteria[1]
            current_node.gain = best_gain
            current_node.tb = DecisionNode()
            current_node.fb = DecisionNode()
            queue.append((current_node.tb, best_sets[0]))
            queue.append((current_node.fb, best_sets[1]))
        else:
            current_node.results = unique_counts(current_part)
    return root


def classify(tree, values):
    if tree.results is not None:
        return list(tree.results.keys())[0]
    else:
        
        if isinstance(values[tree.col], (int, float)):
            split_function = _split_numeric
        else:
            split_function = _split_categorical

        if split_function(values, tree.col, tree.value):
            return classify(tree.tb, values)
        else:
            return classify(tree.fb, values)

    
def prune(tree: DecisionNode, threshold: float) :
    if tree.col == -1:
        return DecisionNode(results=tree.results)

    tree_false = prune(tree.fb, threshold)
    tree_true = prune(tree.tb, threshold)
    
    if tree_true.results is not None and tree_false.results is not None:
        if tree.gain < threshold:
            new_result = tree_true.results.copy()
            new_result.update(tree_false.results)
            return DecisionNode(results = new_result)
    return DecisionNode(col=tree.col, value=tree.value, tb=tree_true, fb=tree_false, gain=tree.gain)
    

def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"

    header, data = read(filename)
    # print_data(header, data)

    # print(unique_counts(data))

    # print(gini_impurity(data))
    # print(gini_impurity([]))
    # print(gini_impurity([data[0]]))

    # print(entropy(data))
    # print(entropy([]))
    # print(entropy([data[0]]))

    # print_data(header, data)

    # part_T, part_F = divideset(data, column=2, value="yes")
    # print_data(header, part_T)
    # print_data(header, part_F)

    # headers, data = read(filename)
    # tree = buildtree(data, gini_impurity, 0.11)
    # print_tree(tree, header)
    # result = classify(tree, data[1])
    # print(result)
    # print_data(header, [data[1]])

    # print("\n------------------------------------------\n")
    # new_tree = prune(tree, 0.4)
    # print_tree(new_tree, header)


if __name__ == "__main__":
    main()