import pandas as pd

from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(DecisionTreeClassifier):
    def __init__(self, features=None, classes=None, max_depth=None):
        super().__init__(max_depth=max_depth)
        self.feature_names = features
        self.class_names = classes
   
    def fit(self, X, y):
        super().fit(X, y)
        
     
    def graph(self, feature_names=None, class_names=None):
        dot_data = export_graphviz(self, out_file=None, 
                                   feature_names=self.feature_names,  
                                   class_names=self.class_names,  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        graph = graphviz.Source(dot_data)  
        return graph
    
    
file = 'train.csv'
X = pd.read_csv(file) # load labeled dataset (with a class)
X = X.set_index('TransactionID')  # use as index and drop from table 

for A in X:
    X = X[X[A] != 'NotFound'] # todo instead use most common value or closest row

#X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42)


# only numeric and small for testing
y = X.iloc[:200,-1] 
X = X.iloc[:200,9:-1] 

print(X.head())

tree = DecisionTree(X.columns, y.unique().astype(str))
tree.fit(X.to_numpy(), y.to_numpy())
dot = tree.graph()
dot.render(filename='dot', format='png')
