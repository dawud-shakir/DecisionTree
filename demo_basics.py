import numpy as np
import pandas as pd
import os
import time

### dataset ##########################

categorical =  (
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2"
)

continuous = (  
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
)

target = (
    "isFraud"
)

### accuracy ############################################################

def imbalance_ratio(y:pd.Series):
    counts = y.value_counts()
    minor = counts[1]
    major = counts[0]
    return float(minor / major)


def score(y, predictions):
    if len(y) != len(predictions):
        exit(f'y and predicted are different lengths: [{len(y)}, {len(predictions)}]')

    y = list(y)
    predictions = list(predictions)
    

    n_correct_0 = n_wrong_0 = n_wrong_1 = n_correct_1 = 0

    # calculate confusion matrix values
    for i in range(len(y)):
        if y[i] == predictions[i]:
            if y[i] == 0:
                n_correct_0 += 1
            else:
                n_correct_1 += 1
        else: 
            if y[i] == 0:
                n_wrong_0 += 1
            else:
                n_wrong_1 += 1

    return n_correct_0, n_wrong_0, n_correct_1, n_wrong_1


def confusion_matrix(y, predictions):
    correct_0, wrong_0, correct_1, wrong_1 = score(y, predictions)
    return [[correct_0, wrong_0], [wrong_1, correct_1]]
    

def accuracy(y, predictions):

    correct_0, wrong_0, correct_1, wrong_1 = score(y, predictions)
    N = correct_0 + wrong_0 + wrong_1 + correct_1

    accuracy = 1 - ((wrong_0 + wrong_1) / N)
    weighted_accuracy = 1 - (0 * correct_0 + (1 * wrong_0 + imbalance_ratio(y) * wrong_1 + 0 * correct_1) / N) # ignore correct
    
    return accuracy, weighted_accuracy

def standard_error(y, predictions):
    acc, weighted_accuracy = accuracy(y, predictions)
    N = len(y)
    
    standard_error = np.sqrt((1 / N) * acc * (1 - acc))
    weighted_error = np.sqrt((1 / N) * weighted_accuracy * (1 - weighted_accuracy))

    return standard_error, weighted_error



### missing values ###############################################################

def remove_iterative(df:pd.DataFrame, columns_to_remove):
    print(f"removing iterative columns {columns_to_remove}")
    df = df.drop(columns_to_remove, axis=1)
    return df

def impute_missing_data(df:pd.DataFrame):
    print("imputing missing data")

    # use not-a-number representation that is compatible with pandas
    df = df.replace(['NotFound', 'NaN'], float('nan')) 
    

    nan_count_per_column = df.isna().sum()
    nan_count_total = df.isna().sum().sum()


                         
    print(f"imputing {nan_count_total} values")

    # impute missing values with most common value in column
    for i in df:
        df[i] = df[i].fillna(df[i].mode()[0])



    return df

def drop_missing_data(df):
    print("dropping missing data")
    # use not-a-number representation that is compatible with pandas
    df = df.replace(['NotFound', 'NaN'], float('nan')) 

    # drop rows
    df = df.dropna(axis=0)
    return df


### impurity ##############################################

def purity(df_test, column_name):
    target_column = df_test[column_name]
    unique_classes = np.unique(target_column)
    if int(len(unique_classes))==1:
        return True
    else:
        return False
    

def gini_index(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts/len(y)
    gini_i = 1 - np.sum(probabilities**2)
    return gini_i


def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-10)))
    
    return entropy

### splitters #########################################################

def best_split_threshold(df, column, impurity):


    if False:
        # brute force
        orderedvals = df[column].sort_values().values
    
    #percentiles
    prctls = df[column].describe(percentiles=np.array(range(1,100))/100)

     # exclude count, std, and mean
    orderedvals = [prctls[i] for i in prctls.index if ~np.isin(i, ['count','mean','std'])]


    min_split = float('inf')
    best_valsmids = None

    for i in range(len(orderedvals)-1):
        if False:
            # with mean-stepping!
            valsmids = (orderedvals[i] + orderedvals[i+1])/2
        
        valsmids = orderedvals[i]
        
        databelow = df.loc[df[column] <= valsmids, column]
        dataabove = df.loc[df[column] > valsmids, column]
        
        B = impurity(databelow)
        A = impurity(dataabove)
        
        split = (len(databelow)/(len(databelow) +len(dataabove)))*B + (len(dataabove)/(len(databelow)+len(dataabove)))*A
        
        if split < min_split:
            min_split = split
            best_valsmids = valsmids
    
    return best_valsmids


def numerical_split(df, column, best_valsmids):
    splits = {}
    
    databelow = df.loc[df[column] <= best_valsmids, column]
    dataabove = df.loc[df[column] > best_valsmids, column]

    splits['databelow'] = databelow
    splits['dataabove'] = dataabove

    return splits



#splits = numerical_split(df_test, 'card1', 12000)
#print(splits['databelow'])
 
 
def categorical_split(df, column):
    featurevals = df[column]
    branches = {}
    
    for i in featurevals.unique():
        branches[i] = df.loc[df[column] == i, column]
    
    return branches
        
    

def calc_information_gain(impurity, column, branches, databelow, dataabove):
    
    if branches == None:
        information_gain = impurity(column) - impurity(databelow) - impurity(dataabove)
    
    else:
        information_gain = impurity(column) - np.sum(impurity(branches))
        
        
    return information_gain


def find_column_split_gains(df, impurity_method):
    if df.shape[1] == 1:
        return df # only one column

    timer_start = time.time()
    gains = []

  
    # calculate gains for each column (including TransactionID and isFraud!) 
    for column in df:
        if column in categorical:

            branches = categorical_split(df, column)
            information_gain = calc_information_gain(impurity_method, column, branches, None, None)
            
        else:
            best_valsmids = best_split_threshold(df, column, impurity_method)
            splits =  numerical_split(df, column, best_valsmids)
            information_gain = calc_information_gain(impurity_method, column, None, splits['databelow'], splits['dataabove'])         
                    
        gains.append(information_gain)    

    best_gain = df.columns[np.argmin(gains)]
    print(f"best gain={best_gain}")
    timer_end = time.time()

    print(f"time to calculate {impurity_method.__name__} gain for {len(df)} rows/samples was {timer_end-timer_start}")

    return gains

### ####################################################################


class DecisionNode:
        max_depth = 1

        def __init__(self):
            self.chance_of_1 = {}    # { value: float } 
            self.branches = {}            # { value: None | Node }
                    
        def fit(self, column, y):
            if column.empty: # or any stopping criteria ... 
                return None

            for value in column.unique():
                y_value = y[column == value]
                
                self.chance_of_1[value] = sum(y_value) / len(y_value)
                
                self.branches[value] = None # guaranteed to be 0, or guaranteed to be 1

                if DecisionNode.max_depth > 1:
                     if self.chance_of_1[value] != 0.0 and self.chance_of_1[value] != 1.0:
                
                        child_node = DecisionNode()
                        child_node.fit(column[column == value], y[column == value])
                        self.branches[value] = child_node

            return self

        def predict(self, value):
            if value not in self.branches:
                return None # value not encountered in training
            
            chance_of_1 = self.chance_of_1[value] 
            child_node = self.branches[value]
            if child_node == None:
                return chance_of_1  # this is a leaf...
            else:
                return chance_of_1 * child_node.predict(value)


def build_and_predict(df, column):
    exit("not implemented yet")


# from class ...
def adjust(n_zeros, n_ones):
    minor = min(n_ones,n_zeros)
    major = max(n_ones,n_zeros)
    imbalance_rtaio = imbalance_ratio(minor,major)
    
    weighted_total = (major * imbalance_rtaio) + minor # W_TOTAL = X0*IR + X1 
    
    if weighted_total == 0:
        exit('divide by zero: ir=',imbalance_rtaio,'minor=',minor,' major=',major)
    
    weighted_major = major * (imbalance_rtaio / weighted_total) # X0_new = X0*IR / W_TOTAL
    weighted_minor = minor / weighted_total
    return (weighted_minor, weighted_major)    


# omitted returning weighted_total ... can't remember why it was needed?
#    return (weighted_total, weighted_minor, weighted_major)    
    
 
