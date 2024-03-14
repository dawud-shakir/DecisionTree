# cs529
# calculate and show gains for csv table
# with entropy, misclassification error and gini index

import numpy as np
import pandas as pd

from time import time


categories = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2"
]

continuous = [
    "TransactionDT",
    "TransactionAmt",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14"
]

iterative = ["TransactionID"]

target = "isFraud"

df_train = pd.read_csv("train.csv")



def impurity_of(Y, with_measure) -> float:
    
    if with_measure not in ["entropy", "misclassification_error", "gini_index"]:
        exit("method", with_measure, "not recognized")


    
    value_counts = Y.value_counts()

    probabilities = value_counts / len(Y)

    impurity = None
    
    if with_measure == "entropy":
        probabilities = probabilities[probabilities != 0]
        impurity = -sum(probabilities * np.log2(probabilities)) 


    elif with_measure == "misclassification_error":
        impurity = 1.0 - np.max(probabilities)

    
    elif with_measure == "gini_index":
        impurity = 1.0 - sum(probabilities**2)      

    return impurity

    
# Find gain for a single column.
def find_column_gain(df, column, with_measure) -> float:

    if column == target:
        exit("column == target_column: no predictive information.")
    
    X = df[column]
    Y = df[target]

    Y_impurity = impurity_of(Y, with_measure)  # impurity of all column's values

    if column in categories:
        X_impurity = 0
        unique_values = X.unique()
        for value in unique_values:  
            y_subset = Y[X == value]
            X_impurity += (len(y_subset) / len(Y)) * impurity_of(y_subset, with_measure)

        gain = Y_impurity - X_impurity

    else:
        threshold = X.median()
        y_subset_below = Y[X <= threshold]
        y_subset_above = Y[X > threshold]
        
        X_impurity = (len(y_subset_below) / len(Y)) * impurity_of(y_subset_below, with_measure) + \
                     (len(y_subset_above) / len(Y)) * impurity_of(y_subset_above, with_measure)
       
        gain = Y_impurity - X_impurity

    in_range = None

    if with_measure == "entropy":

        in_range = gain >= 0 and gain <= np.log2(Y.nunique())
    
    elif with_measure == "misclassification_error":

        in_range = gain >= 0 and gain <= 1.0

    elif with_measure == "gini_index":
        
        in_range = gain >= 0 and gain <= 1.0

    if in_range != True:
        # round-off?
        pass

    return gain


start_time = time()

with_measure = "entropy"
gains = {}

for i in df_train.columns:
    if i != target:
        entropy = find_column_gain(df_train, i, with_measure)
        gains[i] = entropy

print("+"*15, "gains using", with_measure, "+"*15)
gains = pd.Series(gains)

print(gains)
print("sum=", sum(gains))



with_measure = "misclassification_error"
gains = {}

for i in df_train.columns:
    if i != target:
        entropy = find_column_gain(df_train, i, with_measure)
        gains[i] = entropy

print("+"*15, "gains using", with_measure, "+"*15)
gains = pd.Series(gains)
print(gains)
print("sum=", sum(gains))



with_measure = "gini_index"
gains = {}

for i in df_train.columns:
    if i != target:
        entropy = find_column_gain(df_train, i, with_measure)
        gains[i] = entropy

print("+"*15, "gains using", with_measure, "+"*15)
gains = pd.Series(gains)
print(gains)
print("sum=", sum(gains))


end_time = time()
total_time = end_time - start_time

print("average time: %.4f" % (total_time/3))
