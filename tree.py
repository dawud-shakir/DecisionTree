import numpy as np
import pandas as pd











import os # for path to project directory

from time import time # for performance timing






from sklearn.utils import resample
from sklearn.model_selection import train_test_split







def accuracy(y, predictions):
    # Check if y and predictions have the same size
    if len(y) != len(predictions):
        exit(f'Vectors y and predicted are different sizes: [{len(y)}, {len(predictions)}]')

    # Confusion matrix
    correct_0 = wrong_0 = wrong_1 = correct_1 = 0

    # Calculate confusion matrix values
    for i in range(len(y)):
        if y[i] == predictions[i]:
            if y[i] == 0:
                correct_0 += 1
            else:
                correct_1 += 1
        else: 
            if y[i] == 0:
                wrong_0 += 1
            else:
                wrong_1 += 1

    N = correct_0 + wrong_0 + wrong_1 + correct_1

    # Calculate imbalance ratio, accuracy, weighted accuracy, and standard error
    _,counts = np.unique(y, return_counts=True)
    imbalance_ratio = min(counts) / max(counts)
    accuracy = 1 - ((wrong_0 + wrong_1) / N)
    weighted_accuracy = 1 - (0 * correct_0 + (1 * wrong_0 + imbalance_ratio * wrong_1 + 0 * correct_1) / N) # ignore correct
    standard_error = np.sqrt((1 / N) * accuracy * (1 - accuracy))

    return ([[correct_0, wrong_0], [wrong_1, correct_1]], imbalance_ratio, accuracy, weighted_accuracy, standard_error)



def entropy(x:np.array) -> np.array:
    x = x[x != 0]
    return -np.sum(x * np.log2(x))



'''

Idea: Use category (nominal) as threshold

Assign a percentage to each category

The threshold with highest gain is positioned at 0:

...-2, -1, 0, +1, +2...
'''
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
    return x

def gini_index(target):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts/len(target)
    gini_i = 1 - np.sum(probabilities**2)
    return gini_i



def weighted_entropy(x, y):
    
    entropy = 0
    for v in x.unique():

        subset = y[x == v]

        count_1 = sum(subset) 
        count_0 = len(subset) - count_1
    
        expected_1 = (count_1 / len(subset)) * np.log2(count_1 / len(subset)) if count_1 > 0 else 0
        weight_1 = (count_1 / len(y)) 
        entropy += weight_1 * expected_1
        
        expected_0 = (count_0 / len(subset)) * np.log2(count_0 / len(subset)) if count_0 > 0 else 0
        weight_0 = (count_0 / len(y)) 
        entropy += weight_0 * expected_0
        
    return entropy


def find_best_feature(X, y, impurity_method):
    if X.shape[1] == 1:
        return X.columns[0] # only one column

    gains = []
    for column in X:
         impurity = impurity_method(X[column], y)
         gains.append(impurity)
         #print(f'{column} has {impurity} impurity')

    return X.columns[np.argmax(gains)]

      

class DecisionTree:
    
    class DecisionNode:

        def __init__(self, name):
            self.name = name
            
            
            self.chance_of_1 = dict     # { value: float } 
            self.branch = dict            # { value: None | Node } 

        
        def find_y(self, sample):
            node = self
            

            while 1:
                    
                column = node.name
                if column in sample:
                    value = sample[column]

                    if value in node.leaf_1:
                        return 1
                    elif value in node.leaf_0:
                        return 0
                    else:
                        for subnode_name in node.branch.keys():
                            if value in node.chance_of_1:
                                node = node.branch[subnode_name] # go to next node
                else:            
                    # not found 
                    return None

        def add_value(self, value, target):
            if target == "0":
                self.leaf_0.add(value)
            elif target == "1":
                self.leaf_1.add(value)
            else:
                exit("impossible error in add_value")
                
        def add_branch_to_node(self, edge_name, node):
            if node.name not in self.chance_of_1:
                self.chance_of_1[node.name] = set()
                self.branch[node.name] = node
            self.chance_of_1[node.name].add(edge_name)         
        ### End of DecisionNode #########################################

    def __init__(self, max_depth = None):
        self.root = None
        self.max_depth = max_depth
        


    def build(self, X, y):
        self.root = self.fit(X, y, 0)

    def fit(self, X, y, depth):
        if X.empty or depth == self.max_depth: # stopping criteria or max_samples, min_split, etc...
            return None

        best_column_to_split = find_best_feature(X, y, weighted_entropy) # best_column_to_split
       
        node = DecisionTree.DecisionNode(best_column_to_split)

        column_dropped = X[best_column_to_split]
        X = X.drop(best_column_to_split, axis=1)

        depth += 1 # adding to depth here for good reason!

        for value in column_dropped.unique():
            y_value = y[column_dropped == value]
            
            s = sum(y_value)

            if s == 0:
                # value always 0
                node.add_value(value, "0")
            elif s == len(y_value):
                # value always 1
                node.add_value(value, "1")
            elif X.empty or depth == self.max_depth:
                most_common_value = str(y.mode()[0])
                node.add_value(value, most_common_value)
            else:
                X_branch = X[column_dropped == value]
                branch_node = self.fit(X_branch, y_value, depth)
                node.add_branch_to_node(value, branch_node)
        
        return node
            
        
    def predict(self, X_test):
        predictions = []
        for i,row in X_test.iterrows():
            prediction = self.root.find_y(row)
            predictions.append(prediction)
        
        return predictions
    

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

def preprocess(filepath):
    # load data
    df =  pd.read_csv(filepath)

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

    return df


def evaluate_one_tree(filepath, test_size=0.5, max_depth=None):
    ### preprocess tree ##############################
    df = preprocess(filepath)

    X = df.iloc[:,:-1]      # features
    y = df.iloc[:,-1]       # target (isFraud)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=test_size, 
        random_state=0)
    
    n_total = np.array([len(y_train), len(y_test)])
    n_ones = np.array([y_train.sum(), y_test.sum()]) 
    n_zeros = n_total - n_ones
    p_ones = np.floor(100 * n_ones / n_total)
    p_zeros = 100 - p_ones


    print("="*15, " evaluating one-tree ", "="*15)
    print(f"   target (train) total={n_total[0]}, zeros={n_zeros[0]}, ones={n_ones[0]}")
    print(f"   target (test) total={n_total[1]}, zeros={n_zeros[1]}, ones={n_ones[1]}")


    ### train tree ##############################
    decision_tree = DecisionTree()
    print(f"building tree [average time is 234 seconds] ...")
    build_time = time()
    decision_tree.build(X_train, y_train)
    print(f"build took {np.floor((time()-build_time))} seconds")


    ### test tree ##############################

    print(f"predicting [average time is less than 10 seconds] ...")
    predict_time = time()
    predictions = decision_tree.predict(X_test)
    print(f"predicting took {np.floor(time()-predict_time)} seconds")

    accuracy = int(score(y_test.to_list(), predictions) / len(y_test))
    print(f"accuracy: {accuracy}%%")
    print(f"NaN count (predictions)={predictions.count(None)}")


    


evaluate_one_tree(filepath=os.getcwd() + "/train.csv", test_size=0.25)

exit()






def random_forest(self):
    t1 = time()

    # loading data
    df =  pd.read_csv(os.getcwd() + '/data/train.csv')

    # shift TransactionID to be the dataset's index
    df = df.set_index("TransactionID", drop=True)


    # use not-a-number representation that is compatible with pandas
    df = df.replace(['NotFound', 'NaN'], float('nan')) 

    # replace each column's not-a-number with its most common value
    for i in df:
        df[i] = df[i].fillna(df[i].mode()[0]) # mode() does not include nan

    # convert continuous to categorial - 
    # either '<=theta' or '>theta' where theta is ideal threshold 
    for i in df.loc[:, continuous]:
        df[i] = continuous_to_categorical(df[i], df["isFraud"])
        print(f"Converted continuous column {i} to threshold: {df[i].unique()}")


    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    n_trees = 5
    trees = []

    for i in range(n_trees):
        X_subset = resample(X,n_samples=30, shuffle=True, random_state=0)
        y_subset = resample(y,n_samples=30, shuffle=True, random_state=0)
        
        tree = DecisionTree(max_depth=1)
        tree.build(X_subset,y_subset)
        trees.append(tree)

    X_test = resample(X,n_samples=1, shuffle=True, random_state=0)
    y_test = resample(y,n_samples=1, shuffle=True, random_state=0)
            
    predictions = []
    for i in range(n_trees):
        prediction = tree[i].predict(X_test)
        predictions.append(predictions)

    np.mode(predictions)


        




    


    print(f"preprocess time={time()-t1}")



def sklearn_times(test_size=.20, max_depth=None):

    t = time()

    # loading data
    df =  pd.read_csv(os.getcwd() + '/data/train.csv')

    # shift TransactionID to be the dataset's index
    df = df.set_index("TransactionID", drop=True)


    # use not-a-number representation that is compatible with pandas
    df = df.replace(['NotFound', 'NaN'], float('nan')) 

    # replace each column's not-a-number with its most common value
    for i in df:
        df[i] = df[i].fillna(df[i].mode()[0]) # mode() does not include nan

    dummy_t = time()
    for i_col in range(0,len(categorical)):
        df.iloc[:,i_col] = pd.get_dummies(df.iloc[:,i_col])
    print(f"sklearn dummy time={time()-dummy_t}")

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]


    print(f"X shape={X.shape}")
    print(f"y shape={y.shape}")
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    
    print(f"sklearn: preprocess time={time()-t}")

    # Train
    t = time()
    sklearn_tree = DecisionTreeClassifier()
    sklearn_tree.fit(X_train, y_train)
    print(f"sklearn: fit time={time()-t}")

    # Test
    t = time()
    sklearn_predictions = sklearn_tree.predict(X_test)
    print(f"sklearn: predict time={time()-t}")

    # Accuracy
    sklearn_info = accuracy(y_test, sklearn_predictions)
    print(f'sklearn decision tree accuracy: {sklearn_info.accuracy*100}')










n_total = len(df['isFraud'])
n_ones = df['isFraud'].sum() 
n_zeros = n_total - n_ones
p_ones = int(100 * n_ones / n_total)
p_zeros = 100 - p_ones

print(f"y dataset has {n_total} rows total")
print(f"y dataset has {n_zeros} zeros, {n_ones} ones")
print(f"y dataset is {p_zeros}% zeros, {p_ones}% ones")


print(df)

decision_tree = DecisionTree()

X = df.iloc[:,:-1]            # feature dataset
y = df["isFraud"]             # target dataset

# downsample
n_ones = y.sum() 
n_zeros = len(y) - n_ones
'''
X_0, y_0 = resample(
    X[y == 0],
    y[y == 0],
    replace=False,
    n_samples=n_ones,
    random_state=0
)

X = pd.concat([X[y == 1], X_0])
y = pd.concat([y[y == 1], y_0])
'''
p_ones = int(100 * n_ones / len(y))
p_zeros = 100 - p_ones

print(f"after preprocess: y dataset has {n_zeros} zeros, {n_ones} ones")
print(f"after preprocess: y dataset is {p_zeros}% zeros, {p_ones}% ones")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=.00001, 
    #stratify=y,
    #random_state=0,
    #shuffle=True
)

'''
r = np.random.randint(2, size=len(y_test))
r_res = sum(r==y_test)
p_rand = int(100 * r_res / len(y_test))
print(f"should be better than random: {p_rand}") 
'''

decision_tree = DecisionTree()

start_time = time()
decision_tree.build(X_train, y_train)
print(f"time to build: {time()-start_time}")



start_time = time()
predictions = decision_tree.predict(X_test)
print(f"time to predict: {time()-start_time}")

res = pd.DataFrame({'Predictions':predictions, 'Actual':y_test})
res_whole = pd.concat([X_test, res], axis=1)
print(res_whole)

correct = sum(predictions==y_test)
p_correct = 100 * correct / len(y_test)


print(f"Test had {len(predictions)} rows, {predictions.count(0)} were zero, {predictions.count(1)} were one, {predictions.count(None)} rows were not found")
print(f"{int(p_correct)}% correct")

exit()

#fraud_count = df.groupby('isFraud').count()
#print(fraud_count)



print('='*10, ' minimalist test: only using 3 columns! ', '='*10)

decision_tree = DecisionTree()




raw_data = df.loc[:, ['ProductCD', 'card1', 'C1', 'isFraud']]


raw_data['C1'] = continuous_to_nominal(raw_data['C1'], raw_data['isFraud'])


fraud_count = raw_data.groupby('isFraud').count()

print(fraud_count)

print(raw_data)

X = raw_data.iloc[:,:-1]            # feature dataset
y = raw_data['isFraud']             # target dataset




# downsample
n_ones = y.sum() 

X_0, y_0 = resample(
    X[y == 0],
    y[y == 0],
    replace=False,
    n_samples=n_ones,
    random_state=0
)

X = pd.concat([X[y == 1], X_0])
y = pd.concat([y[y == 1], y_0])


n_ones = y.sum()
p_ones = n_ones / len(y)
p_zeros = 1 - p_ones
print(f"p_zeros={p_zeros}, p_ones={p_ones}")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=.2, 
    stratify=y,
    random_state=0,
    shuffle=True
)


p_ones = y_train.sum() / len(y_train)
p_zeros = 1 - p_ones
print(f"train's split := {p_zeros} zeros, {p_ones} ones")



decision_tree = DecisionTree()

start_time = time.time()
decision_tree.build(X_train, y_train)
print(f"time to build: {time.time()-start_time}")

predictions = decision_tree.predict(X_test)
print(f"Test had {len(predictions)} rows, {predictions.count(0)} were zero, {predictions.count(1)} were one, {predictions.count(None)} rows were not found")

'''
decision_tree = DecisionTree()

df_test = df.loc[:, ['ProductCD', 'card1', 'C1', 'isFraud']]
df_test['C1'] = continuous_to_nominal(df_test['C1'], df_test['isFraud'])

print('='*10, ' minimal test: only using 3 columns! ', '='*10)
print(df_test)

y = df_test.pop('isFraud') # get and drop from df



start_time = time.time()
decision_tree.build(df_test, y)
print(f"time to build: {time.time()-start_time}")
'''    
exit()



# Print number of unique values in each columns
X,y=df.iloc[:,:-1],df.iloc[:,-1]
for i in X:
    leafs = 0
    for v in X[i]:
        if y[X[i]==v].nunique()==1:
            leafs += 1
    print(f'{i} has {df[i].nunique()} and {leafs} leafs')
    


for i in df:
    not_found_count = (df[i] == 'NotFound').sum()
    #print(f'column {i} has {not_found_count} NotFound values')
    if not_found_count == 0:
        continue
    elif not_found_count > 0.1*len(df[i]): # comment this line to see the fillna() working for addr1 and addr2
        df = df.drop(i, axis=1)    
        print(f'dropping column {i}')
    else:
        df[i] = df[i].replace('NotFound', np.nan)
        df[i] = df[i].fillna(df[i].mode()[0])


print(df.head())
exit()


#def mini_data(df, fraction):
    #start = int(len(df)*fraction)
    #end = len(df) - start
    
    #return df.iloc[start:end]


#print(df[mini_data(df, 0.8)])


#making a subdata set
n = 10
raw_data = df.head(int(len(df)*(n/100)))
df_test2 = df.sample(n)
#print(df_test)
#print('df_test has', len(df_test), 'rows', 'out of originally', len(df), 'columns')

#print(purity(df_test, 'isFraud'))



