import numpy as np
import pandas as pd
import random
import csv

train_df = pd.read_csv('banks.csv')     # training dataframe
test_df = pd.read_csv('banks-test.csv')     # testing dataframe
testing_data = test_df.values


# The Random Forest
#
#
#
# import the training data, generate the forest and present the prediction
def TrainAndTestRandomForest(train_df, numberOfTrees, percentageOfAttributes, testing_data):
    forest = []
    # generate the forest
    for i in range(numberOfTrees):
        tree = build_tree(train_df.values, percentageOfAttributes)
        forest.append(tree)

    predictions_labels_list = random_forest_predict(testing_data, forest)
    accuracy = get_accuracy(predictions_labels_list, test_df.label)

    write_csv(predictions_labels_list, accuracy, numberOfTrees, percentageOfAttributes)


# CSV file writer
def write_csv(predictions_labels_list, accuracy, numberOfTrees, percentageOfAttributes):
    with open('predictions.csv', 'w', newline='') as csvfile:
        fieldnames = ['numberOfTrees', 'percentageOfAttributes', 'accuracy', 'predictions_labels_list']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'numberOfTrees': str(numberOfTrees), 'percentageOfAttributes': str(percentageOfAttributes),
                         'accuracy': str(accuracy), 'predictions_labels_list': predictions_labels_list})


# predict the testing data in the forest and return the prediction label list
def random_forest_predict(testing_data, forest):
    df_predictions = {}
    # get the prediction result in each tree
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = predictions_collect(testing_data, forest[i])  # return prediction array
        df_predictions[column_name] = predictions
    # vote the best decision in numberOfTrees sets prediction
    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    random_forest_predictions_labels = []
    prediction_labels = np.ndarray.tolist(random_forest_predictions.values)
    # extract the label from the prediction dict to the prediction list
    for i in range(len(prediction_labels)):
        for key in random_forest_predictions[i].keys():
            res = list(random_forest_predictions[i].keys())[0]
            if random_forest_predictions[i][res] < 0.5:
                res = list(random_forest_predictions[i].keys())[1]
            else:
                res = list(random_forest_predictions[i].keys())[0]
        random_forest_predictions_labels.append(res)
    print(random_forest_predictions_labels)

    return random_forest_predictions_labels


# calculate the accuracy of the predictions and the label
def get_accuracy(predictions_labels_list, labels):
    predictions_correct = predictions_labels_list == labels
    accuracy = float(predictions_correct.mean())
    print("accuracy = " + str(accuracy))
    return accuracy


# The Decision Tree:
#
#
#
# Count the number of each class types in current dataset
def class_counts(data_rows):
    counts = {}
    for row in data_rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# split the dataset with match the split point value
def partition(data_rows, split_point):
    true_data_rows, false_data_rows = [], []
    for row in data_rows:
        if split_point.match(row):
            true_data_rows.append(row)
        else:
            false_data_rows.append(row)
    return true_data_rows, false_data_rows


# calculation of the gini impurity
def gini_imp(data_rows):
    total = len(data_rows)
    counts = class_counts(data_rows)
    impurity = 0
    for i in counts.keys():
        p = float(counts[i]) / total
        impurity += p * (1 - p)
    return impurity


# calculation of the info_gain
def get_info_gain(left, right, current_impurity):
    prob = float(len(left)) / (len(left) + len(right))
    info_gain = current_impurity - prob * gini_imp(left) - (1 - prob) * gini_imp(right)
    return info_gain


# get the best split, it determine by the percentageOfAttributes
def find_best_split(data_rows, percentageOfAttributes):
    best_info_gain = 0
    best_split_point = None
    current_impurity = gini_imp(data_rows)

    columns = []
    # the columns contain the random index of the attributes
    while len(columns) < (len(data_rows[0]) - 1) * percentageOfAttributes:
        index = random.randrange(len(data_rows[0]) - 1)
        if index not in columns:
            columns.append(index)

    for index in columns:
        values = set([row[index] for row in data_rows])  # values in this index of column

        for val in values:
            split_point = Split_point(index, val)  # split at this index and value
            true_rows, false_rows = partition(data_rows, split_point)  # partition the dataset with corresponding value
            # end the partition
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            info_gain = get_info_gain(true_rows, false_rows, current_impurity)
            # find the best info_gain and split point
            if info_gain >= best_info_gain:
                best_info_gain = info_gain
                best_split_point = split_point
    return best_info_gain, best_split_point


# build the decision tree
def build_tree(data_rows, percentageOfAttributes):
    info_gain, split_point = find_best_split(data_rows, percentageOfAttributes)
    # no further split, leaf node reached
    if info_gain == 0:
        return Leaf_node(data_rows)

    true_rows, false_rows = partition(data_rows, split_point)
    # recursion the tree, build the true branch and false branch
    true_branch = build_tree(true_rows, percentageOfAttributes)
    false_branch = build_tree(false_rows, percentageOfAttributes)
    # return a split node which record the best split question
    return DecisionNode(split_point, true_branch, false_branch)


# tree print method
def print_tree(node, indentation=""):
    if isinstance(node, Leaf_node):
        print(indentation + "Predict", node.predictions)
        return

    print(indentation + str(node.split_point))

    print(indentation + '-> True:')
    print_tree(node.true_branch, indentation + ' ')

    print(indentation + '-> False:')
    print_tree(node.false_branch, indentation + ' ')


# classify method
def get_classify(current_row, node):
    if isinstance(node, Leaf_node):
        return node.predictions

    if node.split_point.match(current_row):
        return get_classify(current_row, node.true_branch)
    else:
        return get_classify(current_row, node.false_branch)


# get the probability of the number of the classify result, and return the predict result
def get_probability(counts):
    total = sum(counts.values()) * 1.0
    probability = {}

    for label in counts.keys():
        prob = float(counts[label] / total)
        probability[label] = prob

    return probability


# collect the prediction results with input
def predictions_collect(testing_data, mytree):
    predictions_array = []

    for row in testing_data:
        prob = get_probability(get_classify(row, mytree))
        predictions_array.append(prob)

    return predictions_array


# helper classes:
# split point class
class Split_point:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # compare the feature value in an row to the feature value in the split point
    def match(self, row):
        val = row[self.column]
        return val == self.value


# leaf node class, it holds a dict of class
class Leaf_node:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


# decision node class
class DecisionNode:
    def __init__(self, split_point, true_branch, false_branch):
        self.split_point = split_point
        self.true_branch = true_branch
        self.false_branch = false_branch


# Perform the test
TrainAndTestRandomForest(train_df, numberOfTrees=50, percentageOfAttributes=0.5, testing_data=testing_data)

