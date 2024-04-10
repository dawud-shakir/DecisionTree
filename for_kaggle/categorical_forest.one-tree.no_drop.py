import numpy as np
import pandas as pd
import os
import time

import demo_basics as demo

### demo_gains ####################################################################

# load data
df =  pd.read_csv(os.getcwd() + '/train.csv')

forest = []

for i in demo.categorical:
    one_tree = demo.DecisionNode()
    one_tree.fit(df[i], df["isFraud"])
    forest.append(one_tree)

df_test = pd.read_csv(os.getcwd() + '/test.csv')
X_test = df_test

timer_start = time.time()



none_returned = 0
predictions = []
for (_,row) in X_test.iterrows():

    predict_of = []
    for idx,column in enumerate(demo.categorical):
        value = row[column]
        chance_of_1 = one_tree.predict(value)
        if chance_of_1 == None:
            none_returned += 1
            predict_of.append(np.random.randint(2))    # not found
        elif chance_of_1 < 0.5:
            predict_of.append(0)
        else:
            predict_of.append(1)
    vote = sum(predict_of)/len(demo.categorical)
    if vote < 0.5:
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
out_pd.to_csv(os.getcwd() + "/out_forest.csv")








### up to here #######################

