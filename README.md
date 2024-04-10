\documentclass{article}
\usepackage{hyperref}
\usepackage{soul}

\begin{document}

\title{Project 1: Random Forests}
\author{\author{[Caleb Annan, MingChe Pei, Dawud Shakir, Sina Mokhtar]}}
\date{\today}
\maketitle

\section*{Theory discussed}

Report on overleaf (updated last with table): \href{https://www.overleaf.com/project/65e797a9e7782c058ef218cc}{Overleaf Link}


\section*{Overview}

(\texttt{taken verbatum from our report})

Random forests are a collection of small decision trees. They are built using small, randomly selected samples. Forests are discriminatory (not generative) learners and are iteratively built. They become better at predicting after each iteration.

In a forest, each tree is built (or "fitted") with random samples from a dataset. The, one random sample is selected from our dataset for testing and each tree makes one prediction: 0 or 1 (False or True). The forest majority's prediction is compared to the actual value. Samples that cause the majority to give an incorrect value are \textit{more likely} to be chosen in subsequent iterations.

After a number of iterations, the forest majority becomes better at predicting than a single decision tree. 



Each decision tree is built using features (attributes) and a target from the training dataset.  Using Python3 with Pandas, we preprocess the training and testing dataset, we remove columns and subcolumns without meaningful information (iterative columns, $\chi^2$ testing), we replace or drop missing values, we threshold continuous variable data. After preprocessing, we model decision trees individually and collectively (forests), we use hyperparameters to optimize model performance, and evaluate our model with accuracy, precision, recall, and F1-score. 

This repository has several folders: 
\begin{enumerate}

\item \texttt{python-demos} \\
      These are intended to be small and simple, showcasing one (or at the most, two) method. These scripts are designed to perform a specific task or set of tasks related to decision trees and random forests.


\item \texttt{for-kaggle} \\
      These are our top kaggle submissions. 

      
\item \texttt{notebook} \\
        This was \underline{not} in our original submission. It contains Caleb's notebook. He sent me it when we first met as a group and I forgot to include it.
            
\item \texttt{matlab-version} \\
        This was \underline{not} in our original submission (although it was written before the four of us began python).  
      
\end{enumerate}


\section{\textbf{Main Folder}}

\section*{\texttt{matlab-version}}

This folder was \underline{not} in our first submission. (I included it here because I forgot it the first time. This was my first decision tree. I wrote it after reading Mitchell Chapter 03.  In our write up, I discuss its design, more-so than the python version we wrote later. Please ignore if including it would impact anything.)

\begin{itemize}
    \item \textbf{read\_data:} Reads data from a CSV file and preprocesses it.
    \item \textbf{k\_fold:} Splits data into k folds for cross-validation.
    \item \textbf{id3:} Builds a decision tree using the ID3 algorithm.
    \item \textbf{displayTree:} Displays the decision tree.
    \item \textbf{Gini\_Index:} Calculates the Gini index.
    \item \textbf{Misclassification:} Computes the misclassification rate.
    \item \textbf{Entropy:} Calculates the entropy.
    \item \textbf{Gain:} Computes information gain for features.
    \item \textbf{predict:} Predicts the target using a decision tree. 
\end{itemize}

\section*{Functions and Descriptions}

\section*{\texttt{demo-basics.py}}

Main header. Contains most classes and functions for all python demos. 



\begin{enumerate}

\item \texttt{imbalance\_ratio(y:pd.Series)} \\
      \textbf{Summary:} Calculates the imbalance ratio of the target variable.
      
\item \texttt{score(y, predictions)} \\
      \textbf{Summary:} Computes the number of correct and wrong predictions.
      
\item \texttt{confusion\_matrix(y, predictions)} \\
      \textbf{Summary:} Generates the confusion matrix based on actual and predicted values.
      
\item \texttt{accuracy(y, predictions)} \\
      \textbf{Summary:} Calculates the accuracy and weighted accuracy of the predictions.
      
\item \texttt{standard\_error(y, predictions)} \\
      \textbf{Summary:} Computes the standard error and weighted error of the predictions.

\item \texttt{remove\_iterative(df:pd.DataFrame, columns\_to\_remove)} \\
      \textbf{Summary:} Removes iterative columns from the DataFrame.
      
\item \texttt{impute\_missing\_data(df:pd.DataFrame)} \\
      \textbf{Summary:} Imputes missing data in the DataFrame.
      
\item \texttt{drop\_missing\_data(df)} \\
      \textbf{Summary:} Drops rows with missing data from the DataFrame.
      
\item \texttt{purity(df\_test, column\_name)} \\
      \textbf{Summary:} Checks if a column in a DataFrame is pure.
      
\item \texttt{gini\_index(y)} \\
      \textbf{Summary:} Computes the Gini index of a target variable.
      
\item \texttt{calculate\_entropy(y)} \\
      \textbf{Summary:} Calculates the entropy of a target variable.
      
\item \texttt{best\_split\_threshold(df, column, impurity)} \\
      \textbf{Summary:} Finds the best split threshold for a numerical feature.
      
\item \texttt{numerical\_split(df, column, best\_valsmids)} \\
      \textbf{Summary:} Splits a DataFrame based on a numerical threshold.
      
\item \texttt{categorical\_split(df, column)} \\
      \textbf{Summary:} Splits a DataFrame based on a categorical feature.
      
\item \texttt{calc\_information\_gain(impurity, column, branches, databelow, dataabove)} \\
      \textbf{Summary:} Calculates the information gain for a split.
      
\item \texttt{find\_column\_split\_gains(df, impurity\_method)} \\
      \textbf{Summary:} Finds the information gain for each column in the DataFrame.
      
\item \texttt{DecisionNode} \\
      \textbf{Summary:} Class representing a node in a decision tree.
      
\item \texttt{build\_and\_predict(df, column)} \\
      \textbf{Summary:} Builds and predicts using a decision tree.
      
\item \texttt{adjust(n\_zeros, n\_ones)} \\
      \textbf{Summary:} Adjusts the imbalance ratio for a given number of zeros and ones.

\end{enumerate}

\section{\textbf{/python-demo}}


\section*{\texttt{drop-and-split.py}}
Drop "NotFound" and split into train and validating sets

\section*{\texttt{impute-and-split.py}}
Impute missing values and split train and validating sets

\section*{\texttt{one-tree.py}}
Show just one tree train and test

\section*{\texttt{tree-graph.py}}
Show a tree with graphviz

\section*{\texttt{gains.py}}
Displays gains for \textit{some} features



\section*{\texttt{demo-all-gains.py}}
Displays gains for \textit{all} features (iterative included)

    
\section*{\texttt{continuous\_to\_categorical.py}}
Utility to convert continuous data to categorical data

\section*{\texttt{chi\_squared\_signf.py}}
Used to calculate chi-squared and determine significance




\section{\textbf{/for-kaggle}}

\section*{\texttt{demo-70-onetree-nodrop-undersample.py}}

Does not "NotFound" drop with undersampling: 70\% score on leaderboard.


\section*{\texttt{demo-card1.one-tree.no-drop.py}}

Does not "NotFound" drop: 55\%  score on leaderboard.


\subsection*{Usage}
To train our model, begin with our basic template:

\begin{verbatim}
# hello_trees.py

import numpy as np
import pandas as pd
import demo_basics as demo

# Load data
df =  pd.read_csv(os.getcwd() + '/train.csv')

forest = []

for i in demo.categorical:
    one_tree = demo.DecisionNode()
    one_tree.fit(df[i], df["isFraud"])
    forest.append(one_tree)
\end{verbatim}
Then execute the following command:
\begin{verbatim}
python hello_trees.py
\end{verbatim}

\subsection*{Usage}


\begin{enumerate}
    \item To fit our model, use \textit{DecisionNode.fit}:

    \begin{verbatim}
    import demo_basics as demo
    
    # load data
    df =  pd.read_csv(os.getcwd() + '/train.csv')
    
    # build forest
    forest = []
    for i in demo.categorical:
        one_tree = demo.DecisionNode()
        one_tree.fit(df[i], df["isFraud"])
        forest.append(one_tree)
    
    \end{verbatim}
    
    \item To predict with our model, use \textit{DecisionNode.predict}:
    \begin{verbatim}
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

    
    \end{verbatim}

    \item How to create a submission for Kaggle
    \begin{verbatim}
    if 0:
        % for kaggle submission
        start_at = 472433
        out_index = range(start_at, start_at+len(df_test))
        out_predictions = predictions
        out_pd = pd.DataFrame({"TransactionID":out_index, "isFraud":out_predictions})
        out_pd = out_pd.set_index("TransactionID", drop=True)
        out_pd.to_csv(os.getcwd() + "/out_forest.csv")
    end
    
    \end{verbatim}
\end{enumerate}


\section*{Requirements}

Ensure you have the following dependencies installed:

\begin{itemize}
    \item Python (anaconda) version 3.11.5
    \item Additional libraries: NumPy, Pandas, \st{Scikit-learn (for comparing with our from-scratch model}
\end{itemize}

\section*{License}

This project is licensed under the \href{https://opensource.org/licenses/MIT}{MIT License}.

\end{document}
