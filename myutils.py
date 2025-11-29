import numpy as np





# can be helpful to classify based on certain columns 
def make_y_col_lists(my_table, col_name, X_train):
    """
    Creates a training set of y values that is parallel to the X-train set 
    ARGS:
        my_table: the full table that we access column names from 
        col_name: the column that we will be extracting values from 
        X_train: 2D list of values 
    Returns: 
        y_train (1D list) = the training set that we will be using to fit our model 
    """
    col_index = my_table.column_names.index(col_name)
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

        