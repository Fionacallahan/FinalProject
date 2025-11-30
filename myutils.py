import numpy as np
import evaluation


def bin_bmi(value):
    if value < 18.5: return "Underweight"
    elif value < 25: return "Normal"
    elif value < 30: return "Overweight"
    else: return "Obese"

def bin_bp(value):
    if value < 120: return "Normal"
    elif value < 140: return "Elevated"
    elif value < 160: return "Stage1"
    else: return "Stage2"

def bin_glucose(value):
    if value < 100: return "Normal"
    elif value < 126: return "Prediabetes"
    else: return "Diabetes"


def cross_val_predict(X, y, classifier, k=10):
    """
    Runs kfold_split on an X and y set of values. It calculates the accuracy of this method
    Agrs:
        X
        y
        classifier we are evaluating 
        k = 10 (always 10)

    Returns: 
        Accuracy 
        Error_rate
        all_true, all_pred: to be used later in confusion matrices 
    
    """
    folds = evaluation.kfold_split(X, n_splits=k, shuffle=True, random_state=0)
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for train_idexx, test_idex in folds:
        X_train = [X[i] for i in train_idexx]
        y_train = [y[i] for i in train_idexx]
        X_test = [X[i] for i in test_idex]
        y_test = [y[i] for i in test_idex]

        #y_train = [mpg_discretize(y) for y in y_train]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        #y_test = [mpg_discretize(y) for y in y_test]
        #y_pred = [mpg_discretize(p) for p in y_pred]
        
        #FOR CONFUSION MATRIX LATER 
        all_true.extend(y_test)
        all_pred.extend(y_pred)


        correct += sum([pred == true for pred, true in zip(y_pred, y_test)])
        total += len(y_test)

    accuracy = correct / total
    error_rate = 1 - accuracy
    return accuracy, error_rate, all_true, all_pred


# can be helpful to classify based on certain columns 
def make_y_col_lists(header, col_name, X_train):
    """
    Creates a training set of y values that is parallel to the X-train set 
    ARGS:
        my_table: the full table that we access column names from 
        col_name: the column that we will be extracting values from 
        X_train: 2D list of values 
    Returns: 
        y_train (1D list) = the training set that we will be using to fit our model 
    """
    col_index = header.index(col_name)
    y_train = []
    for row in X_train:
        y_train.append(row[col_index])

    return y_train

## pass through if we need to normalize! 
def normalize_column(table, col, little_set):
    """
    Normalizes set of values so that their weight does not mess up the prediction 
    ARGS: 
        table: full table to extract values from 
        col: column we are normalizing 
        little_set: 2D list that contains the values we need to normalize 
    Returns: 
        1D list that contains the normalized values 
    """
    col_index = table.column_names.index(col)
    col_values = [row[col_index] for row in little_set]
    min_val = min(col_values)
    max_val = max(col_values)
    return [[(x - min_val) / (max_val - min_val)] for x in col_values]

        