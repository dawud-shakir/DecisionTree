# cs529 - continuous_to_categorical (entropy)



import sys # for sys.exit(__main__)

import numpy as np
import pandas as pd


def continuous_to_categorical(impurity_method, x, y) -> pd.Series:
    
    
    if x.nunique() <= 5: 
        unique_values = sorted(x.unique()) # brute force

    else:
        x_statistics = x.describe(percentiles=np.array(range(1,100))/100)
        unique_values = [x_statistics[s] for s in x_statistics.index if ~np.isin(s, ['count','mean','std'])] # exclude count, std, and mean

        #unique_values.sort() 

    if len(unique_values) == 1:
        best_threshold = unique_values[0]
        x = np.where(x <= best_threshold, '<=' + str(best_threshold), '>' + str(best_threshold))
        return x

   
    y_entropy = impurity_method(y.value_counts() / len(y))
    gains = []
    gain_ratios = []

    for i in range(0, len(unique_values) - 1): # ignore x_max because it cannot be a threshold
        
        threshold = unique_values[i]

        # lte := less than or equal
        subset_lte = y[x <= threshold]
        lte_entropy = impurity_method(subset_lte.value_counts() / len(subset_lte))
        lte_probability = len(subset_lte) / len(y)
        
        # mt := more than
        subset_mt = y[x > threshold]
        mt_entropy = impurity_method(subset_mt.value_counts() / len(subset_mt))
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
