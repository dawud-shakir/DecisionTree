import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

import random
from pprint import pprint


#loading data
data =  pd.read_csv('C:/users/annan/Onedrive/Desktop/Thon/train.csv')
#print(data.head())

#change 'Notfound' to NaN
#for i in data.columns:
 #   data[i] = data[i].replace('Notfound', 'NaN')




df = pd.DataFrame(data)
df = df.drop('TransactionID', axis=1)
print(df)

#def mini_data(df, fraction):
    #start = int(len(df)*fraction)
    #end = len(df) - start
    
    #return df.iloc[start:end]


#print(df[mini_data(df, 0.8)])


#making a subdata set
n = 1
df_test = df.sample(int(len(df)*(n/100)))
print(df_test)
print('df_test has', len(df_test), 'rows', 'out of originally', len(df), 'columns')



#testing purity of columns

def purity(df_test, column_name):
    target_column = df_test[column_name]
    unique_classes = pd.unique(target_column)
    if int(len(unique_classes))==1:
        return True
    else:
        return False
    
print(purity(df_test, 'isFraud'))

#df_test['ProductCD'].value_counts().plot(marker='o')
#plt.show()


#handling missing data
#print(df[i].isnull().sum().sum()) 

def impute_missing_data(df, column):
        if df[column].index.equals(pd.Index([0])):
            df[column]=df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].index.equals(pd.Index([3])):
            df[column]=df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].index.equals(pd.Index([4])):
            df[column]=df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].index.equals(pd.Index([6])):
            df[column]=df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column]= df[column].fillna(df[column].mean(), inplace=True)
            
        return 'Missing values replaced'
            
      

def gini_index(target):
    unique, counts = pd.unique(target, return_counts=True)
    probabilities = counts/len(target)
    gini_i = 1 - pd.sum(probabilities**2)
    return gini_i

def gini_index_columns(column, target):
    unique, counts = pd.unique(column, return_counts=True)
    for unique in column:
        count_ones = df[(df[column]==unique) & (df[target]==1)]
        count_zeros = df[(df[column]==unique) & (df[target]==0)]
        probability_zeros = count_zeros/count_ones+count_zeros
        probability_ones = count_ones/count_ones+count_zeros


    ginivalue = 1-(probability_ones**2 + probability_zeros**2)

    return ginivalue



def calculate_entropy(column):
        count_ones = df[df[target] == 1].shape[0]
        count_zeros = df[df[target] == 0].shape[0]
        probability_zeros = count_zeros/count_ones+count_zeros
        probability_ones = count_ones/count_ones+count_zeros

    entropyvalue = -(probability_ones*log2(probability_ones) + probability_zeros*log2(probability_zeros))
    
    return entropy


def calculate_entropy_column(column, target):
    unique, counts = pd.unique(column, return_counts=True)
    for unique in column:
        count_ones = df[(df[column]==unique) & (df[target]==1)]
        count_zeros = df[(df[column]==unique) & (df[target]==0)]
        probability_zeros = count_zeros/count_ones+count_zeros
        probability_ones = count_ones/count_ones+count_zeros

    entropyvalue = -(probability_ones*log2(probability_ones) + probability_zeros*log2(probability_zeros))

    return entropyvalue
#print(len(set(df['ProductCD'])))

'''
def best_split_val(df, feature, impurity):
    orderedvals = df[feature].sort_values().values
    min_gini_split = float('inf')
    best_valsmids = None
    
    for i in range(len(orderedvals)-1):
        valsmids = (orderedvals[i] + orderedvals[i+1])/2
        databelow = df['isFraud'][df[feature]] < valsmids
        dataabove = df['isFraud'][df[feature]] >= valsmids
        
        giniB = impurity(databelow)
        giniA = impurity(dataabove)
        
        gini_split = (len(databelow)/(len(databelow) +len(dataabove)))*giniB + (len(dataabove)/(len(databelow)+len(dataabove)))*giniA
        
        if gini_split < min_gini_split:
            min_gini_split = gini_split
            best_valsmids = valsmids
    
    return best_valsmids
'''

def continuous_to_nominal(column:pd.Series, target:pd.Series) -> pd.Series:
    
    
    if column.nunique() <= 5:
        unique_values = sorted(column.unique())

    else:
        x_statistics = column.describe(percentiles=np.array(range(1,100))/100)
        unique_values = [column_statistics[s] for s in column_statistics.index if ~np.isin(s, ['count','mean','std'])] # exclude count, std, and mean

        #unique_values.sort() 

    if len(unique_values) == 1:
        best_threshold = unique_values[0]
        column = np.where(column <= best_threshold, '<=' + str(best_threshold), '>' + str(best_threshold))
        return column

   
    target_entropy = calculate_entropy(target)
    gains = []
    gain_ratios = []

    for i in range(0, len(unique_values) - 1): # ignore x_max because it cannot be a threshold
        
        threshold = unique_values[i]

        # lte := less than or equal
        subset_lte = target[column <= threshold]
        lte_entropy = calculate_entropy_column(subset_lte, target)
        lte_probability = len(subset_lte) / len(target)
        
        # mt := more than
        subset_mt = target[column > threshold]
        mt_entropy = calculate_entropy_column(subset_mt, target)
        mt_probability = len(subset_mt) / len(target)
        
        information_gain = calculate_entropy(column) - (lte_probability * lte_entropy) - (mt_probability * mt_entropy)  
        gains.append(information_gain)

        split_info = -(lte_probability * np.log2(lte_probability)) - (mt_probability * np.log2(mt_probability))
        gain_ratios.append(information_gain / split_info)


    threshold_index = gains.index(max(gains)) 
    #threshold_index = gain_ratios.index(max(gain_ratios))

    best_threshold = unique_values[threshold_index]
    column = np.where(column <= best_threshold, '<=' + str(best_threshold), '>' + str(best_threshold))
    return column


'''
def numerical_split(df_test, feature, best_valsmids):
   splits = {}
    
   for i in range(len(df[feature])-1):
        databelow = df[df[feature] < best_valsmids]
        dataabove = df[df[feature] >= best_valsmids]
        
        splits['databelow'] = databelow
        splits['dataabove'] = dataabove
    
        return splits
'''


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







    
    
#for i in df.columns:
 #   not_found_count = (df[i] == 'NotFound').sum()
  #  if not_found_count == 0:
   #     continue
    #elif not_found_count > 0.1*len(df[i]):
     #   df.drop(i, axis=1)
    #else:
     #   impute_missing_data(df,i)
       #print('need to impute')
        



#decision tree

#print(data.loc[:, ~data.columns.isin(['isFraud'])])


     


    
    
