from scipy.stats import chi2

def chi_squared_stat(attr, target):

    num_rows = attr.shape[0] # rows of splitting attribute
    num_ones = sum(target == 1) # target class (1) 
    num_zeros = num_rows - num_ones # target class (0)
 
    chi_stat = 0.
    
    values = attr.unique()
    for value in values:
        has_value = (attr.values == value) 
        total = sum(has_value) 
        trgt_subset = target.iloc[has_value]
        
        actl_ones = sum(trgt_subset == 1) # actual 1's
        pred_ones = total * (num_ones / num_rows) # predicted 1's
        chi_stat += (actl_ones - pred_ones)**2 / pred_ones

        actl_zeros = total - actl_ones # actual 0's
        pred_zeros = total * (num_zeros / num_rows) # predicted 0's
        chi_stat += (actl_zeros - pred_zeros)**2 / pred_zeros
                
    #    print('value=',value,'num_rows=',num_rows,' num_ones (parent)=',num_ones,'observed_ones=',actl_ones,' observed_zeros=',actl_zeros,'expected_ones=',pred_ones,' expected_zeros=',pred_zeros)
    
    return chi_stat

def is_significant(attr,target,alpha): 
    df = (attr.nunique()-1) * (target.nunique()-1) # degrees of freedom
    if df < 1:
        exit('degrees of freedom less than 1')
   
    p_value = chi2.ppf(1 - alpha, df)
    
    # chi squared critical value
    p_value = chi2.ppf(confidence, df)
    chi_stat =  chi_squared_stat(attr, target)
    return (p_value <= chi_stat,  chi_stat, p_value)
