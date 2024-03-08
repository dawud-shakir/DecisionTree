import numpy as np
import pandas as pd
import os


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

def imbalance_ratio(y):
    counts = y.value_counts()
    return counts.min() / counts.max()

def score(y, predictions):
    if len(y) != len(predictions):
        exit(f"y and predictions are different sizes: [{len(y)}, {len(predictions)}]")

    n_correct = 0
    for i in range(len(y)):
        if y[i] == predictions[i]:
            n_correct += 1

    return n_correct


### impurity #####################################################################


def purity(df_test, column_name):
    target_column = df_test[column_name]
    unique_classes = np.unique(target_column)
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
'''
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
'''            
      

def gini_index(target):
    unique, counts = np.unique(target, return_counts=True)
    probabilities = counts/len(target)
    gini_i = 1 - np.sum(probabilities**2)
    return gini_i


def calculate_entropy(target):
    _, counts = np.unique([df[target]], return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-10)))
    
    return entropy



#print(len(set(df['ProductCD'])))

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


def numerical_split(df_test, feature, best_valsmids):
   splits = {}
    
   for i in range(len(df[feature])-1):
        databelow = df[df[feature] < best_valsmids]
        dataabove = df[df[feature] >= best_valsmids]
        
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



### preprocess ####################################################################

# load data
df =  pd.read_csv(os.'train.csv')

# shift TransactionID to be the dataset's index
df = df.set_index("TransactionID", drop=True)

# use not-a-number representation that is compatible with pandas
df = df.replace(['NotFound', 'NaN'], float('nan')) 

'''
# drop and impute missing values
for i in df:
    n_missing = len(df[i]) - df[i].count() 
    
    # remove columns that are more than 90% missing values
    if n_missing > 0.1*len(df[i]):
        df = df.drop(i, axis=1)    
    else:
        df[i] = df[i].fillna(df[i].mode()[0])

    # count() and mode() does not include not-a-number values
'''    

# replace each column's not-a-number with its most common value
for i in df:
    df[i] = df[i].fillna(df[i].mode()[0])

# convert continuous to categorial - 
# all values in column become either '<= threshold' or '> threshold' 
y = df["isFraud"]
for i in df.loc[:, continuous]:
    df[i] = continuous_to_categorical(df[i], y)
    print(f"continuous column {i} converted to categorical: {df[i].unique()}")


print("\n")



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
         if column in 
         impurity = impurity_method(X[column], y)
         gains.append(impurity)

    return X.columns[np.argmax(gains)]

      


def continuous_to_categorical(x:pd.Series, y:pd.Series, impurity=entropy) -> pd.Series:
    if x.nunique() <= 5: 
        # brute force thresholding tests all values
        unique_values = sorted(x.unique())

    else:
        # percentile 1st, 2nd, ..., 100th thresholding tests percentiles/quadrants   
        x_statistics = x.describe(percentiles=np.array(range(1,100))/100)
        unique_values = [x_statistics[s] for s in x_statistics.index if ~np.isin(s, ['count','mean','std'])] # exclude count, std, and mean

        #unique_values.sort() 

    if len(unique_values) == 1:
        best_threshold = unique_values[0]
        x = np.where(x <= best_threshold, '<=' + str(best_threshold), '>' + str(best_threshold))
        return x

   
    y_entropy = entropy(y.value_counts() / len(y))
    gains = []
    gain_ratios = []

    for i in range(0, len(unique_values) - 1): # ignore x_max because it cannot be a threshold
        
        threshold = unique_values[i]

        # lte := less than or equal
        subset_lte = y[x <= threshold]
        lte_entropy = entropy(subset_lte.value_counts() / len(subset_lte))
        lte_probability = len(subset_lte) / len(y)
        
        # mt := more than
        subset_mt = y[x > threshold]
        mt_entropy = entropy(subset_mt.value_counts() / len(subset_mt))
        mt_probability = len(subset_mt) / len(y)
        
        information_gain = y_entropy - (lte_probability * lte_entropy) - (mt_probability * mt_entropy)  
        gains.append(information_gain)

        split_info = -(lte_probability * np.log2(lte_probability)) - (mt_probability * np.log2(mt_probability))
        gain_ratios.append(information_gain / split_info)


    threshold_index = gains.index(max(gains)) 
    #threshold_index = gain_ratios.index(max(gain_ratios))

    best_threshold = unique_values[threshold_index]
    x = np.where(x <= best_threshold, '<=' + str(best_threshold), '>' + str(best_threshold))
    return x, best_threshold

class DecisionNode:

        def __init__(self, max_depth=None):
            self.column = None

            self.chance_of_1 = dict     # { value: float } 
            self.branches = dict            # { value: None | Node } 

            self.max_depth = max_depth

        def __str__(self):
            return self.column
        
        def fit(self, X, y, depth):
            self.depth = self.depth + 1 
            
            if X.empty or self.depth == self.max_depth: # or any stopping criteria ... 
                return None  # this is a leaf...

            self.column = find_best_column_to_split(X, y, calculate_entropy)
            best_column_to_split = X[self.column]
            X = X.drop(self.column, axis=1)
            
            for value in best_column_to_split.unique():
                y_value = y[best_column_to_split == value]
                
                self.chance_of_1[value] = sum(y_value) / len(y_value)
                self.branches[value] = None
                if self.chance_of_1[value] != 0.0 and self.chance_of_1[value] != 1.0:
                    child_node = DecisionNode()
                    child_node.fit(X[best_column_to_split == value], y[best_column_to_split == value], depth)
                    self.branches[value] = child_node

            return self    

        def predict(self, sample):
            
            value = sample[self.column] 
            chance_of_1 = self.chance_of_1[value] 
            child_node = self.branches[value]
            if child_node == None:
                return chance_of_1  # this is a leaf...
            else:
                return chance_of_1 * child_node.predict(sample)

class DecisionTree:
    
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.root = DecisionNode(self.max_depth)
        self.root.fit(X, y, 0)
     
    def predict(self, samples):
        predictions = []
        for _,sample in samples.iterrows():
            prediction = self.root.find_y(sample)
            predictions.append(prediction)
        
        return predictions
