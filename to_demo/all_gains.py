import numpy as np
import pandas as pd
import os
import time

import demo_basics as demo

### demo_gains ####################################################################

# load data
df =  pd.read_csv(os.getcwd() + '/train.csv')
df = demo.drop_missing_data(df)


timer_start = time.time()
gains = []


impurity_method = demo.calculate_entropy
#impurity_method = demo.calculate_gini_index
#impurity = misclassification_error
print(f"calculating gains with {impurity_method.__name__}")

# calculate gains for each column (including TransactionID and isFraud!) 
for column in df:
    if column in demo.categorical:

        branches = demo.categorical_split(df, column)
        information_gain = demo.calc_information_gain(impurity_method, column, branches, None, None)
        print(information_gain)
    else:
        best_valsmids = demo.best_split_threshold(df, column, impurity_method)
        splits =  demo.numerical_split(df, column, best_valsmids)
        information_gain = demo.calc_information_gain(impurity_method, column, None, splits['databelow'], splits['dataabove'])         
                
    gains.append(information_gain)    

pd_gains = pd.DataFrame({"column":df.columns, "gain":gains})
pd_gains.sort_values("gain")
print(pd_gains)

best_gain = df.columns[np.argmin(gains)]
print(f"best gain={best_gain}")
timer_end = time.time()

print(f"time to calculate {impurity_method.__name__} gain for {len(df)} rows/samples was {timer_end-timer_start}")
### up to here #######################

