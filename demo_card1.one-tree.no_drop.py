import demo_basics as demo
import numpy as np
import pandas as pd

import os
import time

from sklearn.model_selection import train_test_split




train_size = 0.99
column = "card1"

### ####################################################################
# load data
df_train = pd.read_csv(os.getcwd() + '/train.csv')
#df_train = demo.remove_iterative(df_train, ["TransactionID"])
#df_train = demo.drop_missing_data(df_train)

accuracy = []
weighted_accuracy = []
found_accuracy = []


test_size = 1-train_size
print("="*15, "train_size=", int(train_size*len(df_train)), " ", "test_size=", int(test_size*len(df_train)), "="*10)

print("training size=%.2f" % train_size)
print("test size=%.2f" % test_size)

one_tree = demo.DecisionNode()



X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]
X_train, _, y_train, _ = train_test_split(X, y,
                                                    train_size=train_size,
                                                    stratify=y, 
                                                    shuffle=True,
                                                    random_state=0)

timer_start = time.time()
one_tree.fit(X_train[column],y_train)
timer_end = time.time()
print(f"time to fit one-tree node: {timer_end-timer_start}")



df_test = pd.read_csv(os.getcwd() + '/test.csv')
X_test = df_test[column]


timer_start = time.time()
none_returned = 0
predictions = []
for value in X_test:

    
    chance_of_1 = one_tree.predict(value)
    if chance_of_1 == None:
        none_returned += 1
        predictions.append(np.random.randint(2))    # not found
    elif chance_of_1 < 0.5:
        predictions.append(0)
    else:
        predictions.append(1)

timer_end = time.time()
print(f"time to predict with one-tree node: {timer_end-timer_start}")

p_nones = none_returned/len(X_test)
print("none_returned=%.2f" % p_nones)
p_found = (1.0-p_nones)
print("found_accuracy=%.2f" % p_found)        


start_at = 472433
out_index = range(start_at, start_at+len(df_test))
out_predictions = predictions
out_pd = pd.DataFrame({"TransactionID":out_index, "isFraud":out_predictions})
out_pd = out_pd.set_index("TransactionID", drop=True)
out_pd.to_csv(os.getcwd() + "/out_pd.csv")



