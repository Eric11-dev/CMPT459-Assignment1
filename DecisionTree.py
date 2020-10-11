import pandas as pd
import random

df = pd.read_csv('D:\\CMPT459\\Assignment_1_Datasets\\banks.csv')
training_data = df.values
header = list(df.columns)


def unique_values(data, col):
    rows = data
    return set([row[col] for row in rows])


def class_counts(data_rows):
    counts = {}
    for row in data_rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def partition(data_rows, split_point):
    true_rows, false_rows = [], []
    for row in data_rows:
        if split_point.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(data_rows):
    total = len(data_rows)
    counts = class_counts(data_rows)
    imp = 0
    for k1 in counts.keys():
        p1 = float(counts[k1]) / total
        imp += p1 * (1 - p1)
    return imp


def info_gain(left, right, current_uncertainty):
    prob = float(len(left)) / (len(left) + len(right))
    gain = current_uncertainty - prob * gini(left) - (1 - prob) * gini(right)
    return gain


def find_best_split(data_rows, percentageOfAttributes):
    best_gain = 0
    best_split_point = None
    current_uncertainty = gini(data_rows)

    columns = []
    while len(columns) < (len(data_rows[0]) - 1) * percentageOfAttributes:
        index = random.randrange(len(data_rows[0]) - 1)
        if index not in columns:
            columns.append(index)

    for index in columns:
        values = set([row[index] for row in data_rows])

        for val in values:
            split_point = Split_point(index, val)
            true_rows, false_rows = partition(data_rows, split_point)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain = gain
                best_split_point = split_point
    return best_gain, best_split_point


def build_tree(data_rows, percentageOfAttributes):
    gain, split_point = find_best_split(data_rows, percentageOfAttributes)

    if gain == 0:
        return Leaf_node(data_rows)

    true_rows, false_rows = partition(data_rows, split_point)

    true_branch = build_tree(true_rows, percentageOfAttributes)

    false_branch = build_tree(false_rows, percentageOfAttributes)

    return DecisionNode(split_point, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf_node):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.split_point))

    print(spacing + '-> True:')
    print_tree(node.true_branch, spacing + ' ')

    print(spacing + '-> False:')
    print_tree(node.false_branch, spacing + ' ')


def get_classify(current_row, node):
    if isinstance(node, Leaf_node):
        return node.predictions

    if node.split_point.match(current_row):
        return get_classify(current_row, node.true_branch)
    else:
        return get_classify(current_row, node.false_branch)


def get_probability(counts):
    total = sum(counts.values()) * 1.0
    probability = {}

    for label in counts.keys():
        prob = float(counts[label] / total)
        probability[label] = prob

    return probability


def predictions_collect(testing_data, mytree):
    predictions_array = []

    for row in testing_data:
        prob = get_probability(get_classify(row, mytree))
        predictions_array.append(prob)

    return predictions_array


class Split_point:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val == self.value

    # def __repr__(self):
    #     condition = "=="
    #     if is_numeric(self.value):
    #         condition = ">="
    #     return "Is %s %s %s?" % (
    #         header[self.column], condition, str(self.value))


class Leaf_node:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class DecisionNode:
    def __init__(self, split_point, true_branch, false_branch):
        self.split_point = split_point
        self.true_branch = true_branch
        self.false_branch = false_branch
