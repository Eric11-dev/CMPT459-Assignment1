import csv
import random
from random import random

import pandas as pd

if __name__ == '__main__':


    df = pd.read_csv('D:\CMPT459\Assignment_1_Datasets\interviewee.csv')
    print(df)

    test_size = 10
    indices = df.index.tolist()
    test_indices = random(population = indices, k = test_size)
    print(test_indices)


