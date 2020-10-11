import numpy as np
import pandas as pd
import csv

from DecisionTree import build_tree, print_tree, predictions_collect

train_df = pd.read_csv('D:\\CMPT459\\Assignment_1_Datasets\\banks.csv')
test_df = pd.read_csv('D:\\CMPT459\\Assignment_1_Datasets\\banks.csv')
testing_data = test_df.values


def TrainAndTestRandomForest(train_df, numberOfTrees, percentageOfAttributes, testing_data):
    forest = []

    for i in range(numberOfTrees):

        tree = build_tree(train_df.values, percentageOfAttributes)
        forest.append(tree)

    predictions_labels_list = random_forest_predict(testing_data, forest)
    accuracy = get_accuracy(predictions_labels_list, test_df.label)

    write_csv(predictions_labels_list, accuracy, numberOfTrees, percentageOfAttributes)


def write_csv(predictions_labels_list, accuracy, numberOfTrees, percentageOfAttributes):

    with open('predictions.csv', 'w', newline='') as csvfile:
        fieldnames = ['numberOfTrees', 'percentageOfAttributes', 'accuracy', 'predictions_labels_list']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow( {'numberOfTrees': str(numberOfTrees), 'percentageOfAttributes': str(percentageOfAttributes),
                          'accuracy': str(accuracy), 'predictions_labels_list': predictions_labels_list})


def random_forest_predict(testing_data, forest):
    df_predictions = {}

    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = predictions_collect(testing_data, forest[i])  # return prediction array
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    random_forest_predictions_labels = []
    prediction_labels = np.ndarray.tolist(random_forest_predictions.values)
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


def get_accuracy(predictions_labels_list, labels):
    predictions_correct = predictions_labels_list == labels
    accuracy = float(predictions_correct.mean())
    print("accuracy = " + str(accuracy))
    return accuracy


TrainAndTestRandomForest(train_df, numberOfTrees=100, percentageOfAttributes=0.6, testing_data=testing_data)
