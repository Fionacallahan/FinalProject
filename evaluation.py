import myutils
import random
import numpy as np

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Computes the precision (for binary classification). Basically the ability of the classifier 
    to label a false positive. The best value: 1 and the worst value: 0

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class
    """
    # best value: 1, worst is zero 
    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == pos_label:
                true_positive += 1
        else:
            if y_pred[i] == pos_label:
                false_positive += 1

    if ((true_positive + false_positive) != 0):
        precision = true_positive/(true_positive + false_positive)

    else:
        precision = 0
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification).  Ability of classifier to find all positive samples

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class
    """
    #similar to precision, but now false negatives 
    true_positive = 0
    false_negative = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == pos_label:
                true_positive += 1
        else:
            if y_pred[i] != pos_label:
                false_negative += 1

    if ((true_positive + false_negative) != 0):
        recall = true_positive/(true_positive + false_negative)

    else:
        recall = 0
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification)
    Arguably, the most helpful

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive classl
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall != 0:

        f1 = 2 * (precision * recall) / (precision + recall)

    else:
        f1 = 0
    return f1


# from previous PAS 
def train_test_split(X, y, test_size=0.33, random_state=0, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    """
    import math # for acquiring test samples

    #shuffle
    if shuffle is True:
        np.random.seed(random_state)
        indices = list(range(len(X)))
        random.shuffle(indices)

        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

    #print(len(X))
    if isinstance(test_size, float):
        #print("in test_size")
        total_samples = len(X)
        #print(total_samples * test_size)
        #print(test_size)
        test_samples = math.ceil(total_samples * test_size)
        #print(test_samples)
    else: 
        test_samples = test_size

    #print(test_samples)

    X = list(zip(X,y))
    train_set = X[:-test_samples] #had to mess with indexing
    test_set = X[-test_samples:]
    X_train, y_train = zip(*train_set) #zipping it together: from notes
    X_test, y_test = zip(*test_set)

    X_train = list(X_train)
    y_train = list(y_train)
    X_test = list(X_test) 
    y_test = list(y_test)

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=0, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
    """
    k = n_splits

    #shuffle
    indices = list(range(len(X)))
    if shuffle:
        #simpler than last time
        np.random.seed(random_state)
        random.shuffle(indices)


    #splitting the data 
    first_size = len(X) // n_splits
    extras = len(X) % n_splits

    fold_sizes = []
    for i in range(n_splits):
        if i < extras:
            fold_sizes.append(first_size + 1)
        else:
            fold_sizes.append(first_size)

    # Now build the folds
    folds = []
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:] #just in case test indices is in middle 
        folds.append((train_indices, test_indices))
        start = end #starts over
    return folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    """

    # set up for visualization
    matrix = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            row.append(0)
        matrix.append(row)


    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        # Find the index of the true label (by index)
        true_index = -1
        for j in range(len(labels)):
            if labels[j] == true_label:
                #gives us row index in matrix 
                true_index = j
                break

        # Find the index of the predicted label (by index)
        pred_index = -1
        for j in range(len(labels)):
            if labels[j] == pred_label:
                pred_index = j
                break

        # add one to the count of which column it belongs in 
        matrix[true_index][pred_index] += 1



    return matrix # TODO: fix this

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float)
    """
    true_positive = 0
    #false_positive = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true_positive += 1
    #print(true_positive)
    #print(true_positive/len(y_true))
    if normalize:
        return true_positive/len(y_true)
    else:
        return true_positive




