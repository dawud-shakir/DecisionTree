import numpy as np
import pandas as pd
import os
import time


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

### evaluation ###############################################################


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
 
 
def categorical_split(df, feature):
    featurevals = df[feature]
    branches = {}
    
    for i in pd.unique(featurevals):
        branches[i] = df[df[feature] == i]
    
    return branches
        
    

def calc_information_gain(impurity, feature, branches, databelow, dataabove):
    
    if branches == None:
        information_gain = impurity(feature) - impurity(databelow) - impurity(dataabove)
    
    else:
        information_gain = impurity(feature) - np.sum(impurity(branches))
        
        
    return information_gain








def gini_index(target):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts/len(target)
    gini_i = 1 - np.sum(probabilities**2)
    return gini_i


def calculate_entropy(target):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-10)))
    
    return entropy


def find_best_column_to_split(X, y, impurity_method):
    if X.shape[1] == 1:
        return X # only one column

    gains = []
    for column in X:
         impurity = impurity_method(X[column], y)
         gains.append(impurity)

    return X.columns[np.argmax(gains)]

      

### ####################################################################

# load data
df =  pd.read_csv(os.getcwd() + '/train.csv')
df = impute_missing_data(df)


timer_start = time.time()
gains = []

#impurity = calculate_entropy
impurity = gini_index
#impurity = misclassification_error

# calculate gains for each column (including TransactionID and isFraud!) 
for column in df:
    if column in categorical:

        branches = categorical_split(df, column)
        information_gain = calc_information_gain(impurity, column, branches, None, None)
        
    else:
        best_valsmids = best_split_threshold(df, column, impurity)
        splits =  numerical_split(df, column, best_valsmids)
        information_gain = calc_information_gain(impurity, column, None, splits['databelow'], splits['dataabove'])         
                
    gains.append(information_gain)    

best_gain = df.columns[np.argmax(gains)]
print(f"best gain={best_gain}")
timer_end = time.time()

print(f"time to calculate {impurity.__name__} gain for {len(df)} rows/samples was {timer_end-timer_start}")
### up to here #######################

