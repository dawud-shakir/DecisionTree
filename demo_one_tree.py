import demo_basics as demo
import numpy as np
import pandas as pd

import os
import time

from sklearn.model_selection import train_test_split




train_size = 0.75
test_size = 1-train_size
column = "card1"

### ####################################################################
# load data
df = pd.read_csv(os.getcwd() + '/train.csv')
df = demo.remove_iterative(df, ["TransactionID"])
df = demo.drop_missing_data(df)


one_tree = demo.DecisionNode()



X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=train_size,
                                                    stratify=y, 
                                                    shuffle=True,
                                                    random_state=0)
print("training size=%.2f" % train_size)
print("test size=%.2f" % test_size)

print(f"number of samples for training={len(X_train)}")
print(f"number of samples for testing={len(X_test)}")

timer_start = time.time()
one_tree.fit(X_train[column],y_train)
timer_end = time.time()
print(f"time to fit one-tree node: {timer_end-timer_start}")



timer_start = time.time()

predictions = []
for value in X_test[column]:

    
    chance_of_1 = one_tree.predict(value)
    if chance_of_1 == None:
        predictions.append(None)    # not found
    elif chance_of_1 < 0.5:
        predictions.append(0)
    else:
        predictions.append(1)

timer_end = time.time()
print(f"time to predict with one-tree node: {timer_end-timer_start}")

n_not_found = predictions.count(None)
n_found =  len(y_test)-n_not_found
p_found = n_found / len(y_test)
accuracy, weighted_accuracy = demo.accuracy(y_test, predictions)

print("accuracy=%.2f" % accuracy)
print("weighted_accuracy=%.2f" % weighted_accuracy)    
print("found_accuracy=%.2f" % p_found)        

