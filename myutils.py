import numpy as np
import evaluation
from collections import Counter
import math


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


## for decision trees: def predict_instances(instance, subtree):
def predict_instances(instance, subtree):
    """
    This is acts as a tree traversal function that takes a subtree and traverses it until a Leaf node is found
    This is done recursively so that it follows the tree all the way down 
    
    Args: 
        instance: the passed in testing instance that needs a label to be predicted of 
        subtree: The shrinking tree that allows the function to traverse
    Returns: 
        The predicted label 
    """
    # essentially a tree traversal function! 
    node_type = subtree[0] # starts at beginning 

    if node_type == "Leaf":
        return subtree[1]
    
    elif node_type == "Attribute":
        att_name = subtree[1]

        att_index = int(att_name.replace("att", "")) 
        att_value = instance[att_index]
       #predicts based on missing branch
        for branch in subtree[2:]: 
            if branch[1] == att_value:
                return predict_instances(instance, branch[2])
    # if no match is found: return majority vote 
    else:
        return majority_vote(subtree)


def print_rules(subtree, conditions, attribute_names, class_name):
    """
    Traverses down a decision tree and clearly lays out the rules that the tree follows. 
    
    Args: 
        subtree: a nested list that contains the structure of our decision tree 
        conditions: passed in from the class print_labels, but usally passed in as an empty list at first 
        attribute_names: specific to the tree: helps join rules together 
        class_name: What we are trying to label
    Returns: 
        NA, just prints the rules  
    """
    node_type = subtree[0]
    if node_type == "Leaf":
        # format - IF cond1 AND cond2 THEN class = label
        rule = "IF " + " AND ".join(conditions)
        rule += f" THEN {class_name} = {subtree[1]}"
        print(rule)
    elif node_type == "Attribute":
        att_name = subtree[1]
        att_index = int(att_name.replace("att", ""))
        att_label = attribute_names[att_index]
        for branch in subtree[2:]:
            val = branch[1]
            new_conditions = conditions + [f"{att_label} == {val}"]
            print_rules(branch[2], new_conditions, attribute_names, class_name)

def compute_entropy(instances):
    """
    A helper function of complete_entropy that does the actual math 
    
    Args: 
        instances: the instances that entropy is calculated from 

    Returns: 
        The calculated entropy from the specific equation that we learned in class 
    """

    if not instances:
        return 0
    labels = [row[-1] for row in instances]
    counts = Counter(labels)
    total = len(labels)
    return sum((-count/total) * math.log2(count/total) for count in counts.values())

def complete_entropy(instances, available_atts, header, attribute_domains):
    """
    Looks at each available attribute and decides which attribute is the best to split on 
    
    Args: 
        instances: The training data, either scaled or full out, that we are looking at to split on down
        available_atts: The attributes that have not yet been split on, that can be looked at to compute entropy on 
        header: The attribute names 
        attribute_domains: What each header entails 

    Returns: 
        the selected attribute 
    """
    base_entropy = compute_entropy(instances)
    best_gain = -1
    best_att = None
    #print(attribute_domains)

    #print("Base entropy:", base_entropy)


    for att in available_atts:
        att_index = header.index(att)
        att_domain = attribute_domains[att]

        # Partition by attribute values
        partitions = {val: [] for val in att_domain}
        for row in instances:
            partitions[row[att_index]].append(row)

        # Weighted entropy - second step of entropy 
        weighted_entropy = 0
        for subset in partitions.values():
            if subset: 
                weighted_entropy += (len(subset)/len(instances)) * compute_entropy(subset)


        # wants to get the most information gain 
        gain = base_entropy - weighted_entropy
        if gain > best_gain:
            best_gain = gain
            best_att = att

    return best_att


def partition_instances(instances, selected_att, header, attribute_domains):
    """
    Puts instances into "groups" to help the tdidt algorithm 
    
    Args - 
        instances: The training data, either scaled or full out, that we are looking at to group on 
        selected_att: Selected from the entropy algorithm to make sure we are most efficiently creating our tree 
        header: The attribute names 
        attribute_domains: What each header entails 

    Returns: 
        Partitions 
    """

    att_index  = header.index(selected_att)
    att_domain = attribute_domains[selected_att]
    partitions = {val: [] for val in att_domain}
    for instance in instances:
        partitions[instance[att_index]].append(instance)
    return partitions


def all_same_class(instances):
    """
    Simple helper function that helps compute Case 1 
    
    instances: The training data, either scaled or full out, that we are looking to see if they are all the same class: a leaf node group 

    Returns: 
        True or False  
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    return True

def majority_vote(instances, domain=None):
    """
    Selects the majority class label to be used for Case 2 and Case 3 
    
    instances: The training data, either scaled or full out, that we are looking to see what the majority vote is 
    domain: If not passed in, automatically none, but if so, then it depends on the order 

    Returns: 
        The vote that is decided upon due to majority voting 
    """
    labels = [row[-1] for row in instances]
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [label for label, c in counts.items() if c == max_count]

    if domain:
        for val in domain:
            if val in candidates:
                return val
            
    else:
        return candidates[0]



def tdidt(current_instances, available_atts, header, attribute_domains):
    """
    The main class for the tdidt algorithm. It splits the data into "branches" and keeps pairing down until a leaf node is hit 
    
    Arguments: 
        current_instances: The training data, either scaled or full out, that we are splitting after each recursive call 
        available_atts: The available attributes that are not already integrated into the tree 
        header: The attribute names 
        attribute_domains:  What each header entails 
    """

    #select attribute using entropy: 
    selected_att = complete_entropy(current_instances, available_atts, header, attribute_domains)
    # this attribute is no longer available 
    available_atts.remove(selected_att)

    tree = ["Attribute", selected_att]

    #create partitions 
    partitions = partition_instances(current_instances, selected_att, header, attribute_domains)

    #base cases 

    # sorted keeps the tree good while testing 
    for att_value in sorted(partitions.keys()):
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]

        #All labels are same! 
        if len(att_partition) > 0 and all_same_class(att_partition):
            # keeps leaf format stable 
            leaf = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
            value_subtree.append(leaf)


        elif len(att_partition) > 0 and len(available_atts) == 0:
            # solve with majority vote leaf node
            leaf = ["Leaf", majority_vote(att_partition), len(att_partition), len(current_instances)]
            value_subtree.append(leaf)


        elif len(att_partition) == 0:
            # have to backtrack and replace att node with majority vote leaf node 
            leaf = ["Leaf", majority_vote(current_instances), 0, len(current_instances)]
            value_subtree.append(leaf)

        else:
            subtree = tdidt(att_partition, available_atts.copy(), header, attribute_domains)
            value_subtree.append(subtree)
           
            # now have to append subtree to value_subtree and value_subtree to tree 

        tree.append(value_subtree)
    return tree

        