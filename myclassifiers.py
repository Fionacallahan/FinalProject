import myutils
import numpy as np
from collections import Counter


class MyRandomForestClassifier:
    """Represents a Random Forest Classifier 

    Attributes:
        n_trees: int [represents number of trees the random forest will be combining together to perform ensemble learning]
        max_depth: int [max_depth to create the branches: keeps the work down]
        min_samples_split: int [makes sure the tree can grow relatively deep]
        n_features: int [number of features that the subset is built off of]
        m_trees: int [chosen for predictive purposes after]
    """
    # using N: 10
    # F: 5 
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=5, m_trees=4):
        """
            initializer for randomForest 
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.m_trees = m_trees
        self.trees= []

    def fit(self, X, y):
        """
        Fits a Random Forest Classifier, taking advantage of the decision tree classifier 
        
        Arguments: 
            X: The remainder set training values 
            y: The remainder set testing values 
        """
        tree_accs = []
        for _ in range(self.n_trees):  
            # decision tree classifier made for each number of number_of_trees 
            tree = MyDecisionTreeClassifier()
            # splits into training and validation 
            X_sample, y_sample, X_val, y_val = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)

            # error catching: 
            if len(X_val) > 0:
                acc = tree.score(X_val, y_val)
            else: 
                acc = 0

            tree_accs.append((tree, acc))

            # picking M trees: 
            tree_accs.sort(key=lambda pair: pair[1], reverse=True)
            top_m_pairs = tree_accs[:self.m_trees]

            top_m_trees = []
            for pair in top_m_pairs:
                tree = pair[0]
                top_m_trees.append(tree)

            # Save them
            self.trees = top_m_trees

    def _bootstrap_samples(self, X, y):  
        """
        Splits the Remainder set into different validation and testing sites to put into each tree 
        Happens for each created tree 

        Arguments: 
            X: a training set that is subsetted from the remainder set for each n tree
            y: a testingg set that is subsetted from the remainder set for each n tree
        """
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        train_idx = indices
        val_idx = list(set(range(n_samples)) - set(train_idx))
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        return X_train, y_train, X_val, y_val


    def _most_common_label(self, y):
        """
        Helps with figuring out the most common label from the creation of the trees 
        
        y: the label that we are counting 
        """
        # makes use of the Counter module 
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        predict class for the RandomForestClassifier 

        X: takes an instance, or sets of instances and predicts the classification label
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # change structure: 
        # want all predictions from same sample for different trees in same inner list 
        tree_preds = np.swapaxes(predictions, 0, 1)  # shape: (n_samples, n_trees) - looked at tutorial for this part 
        final_preds = [self._most_common_label(sample_preds) for sample_preds in tree_preds]
        return np.array(final_preds)


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        """
        header_length = len(X_train[0])
        header = []
        for i in range(header_length):
            header.append("att" + str(i))

        # building up the attribute_domains
        attribute_domains = {}
        for att in header:
            attribute_domains[att] = []

        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

        for i in range(len(train)):
            for j in range(header_length):
                if train[i][j] not in attribute_domains[header[j]]:
                    attribute_domains[header[j]].append(train[i][j])
    
        available_attributes = header.copy()
        tree = myutils.tdidt(train, available_attributes, header, attribute_domains)
        self.tree = tree


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for i in range(len(X_test)):
            # for every X in X_test, have to predict based on the instance

            predictions.append(myutils.predict_instances(X_test[i], self.tree))
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(len(self.X_train[0]))]
        myutils.print_rules(self.tree, [], attribute_names, class_name)

    def score(self, X, y):
        """
        Return the accuracy of the classifier on the given test data and labels.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)



class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        conditionals(nested dict where each class goes to lsit of dicts): The conditional probabilities computed for each
            attribute value/label pair in the training set.
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """

        #will be making in form of a list
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """     

        # priors needs to be a dictionary 
        prior_dict = {}
        for i in range(len(y_train)):
            if y_train[i] in prior_dict:
                prior_dict[y_train[i]] += 1
            else:
                prior_dict[y_train[i]] = 1

        for label in prior_dict:
            prior_dict[label] = prior_dict[label] / len(X_train)
        self.priors = prior_dict

        # conditionals needs to be a nested dictionary 
        conditionals_dict = {}
        for i in range(len(X_train)):
            # getting parallel values 
            x = X_train[i]
            y = y_train[i]

            #checking if labeel is in the dict 
            if y not in conditionals_dict:
                # creating list at each label 
                conditionals_dict[y]  = []
                for i in range(len(x)): 
                    #adding dict per "attribute"
                    conditionals_dict[y].append({})
            for j in range(len(x)):
                val = x[j]
                counts = conditionals_dict[y][j]
                if val in counts:
                    counts[val] += 1
                else:
                    counts[val] = 1
        for label in conditionals_dict:
            for feature_index in range(len(conditionals_dict[label])):
                feature_counts = conditionals_dict[label][feature_index]
                total = 0
                for count in feature_counts.values():
                    total += count
                for val in feature_counts: 
                    feature_counts[val] = feature_counts[val] / total


        #print(conditionals_dict)

        self.conditionals = conditionals_dict

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        #print(X_test)
        for j in range(len(X_test)): 
            #goes through each value in X_test
            #print("X_test: ", X_test[j])
            keys = []
            decision = {}
            #through each key
            for key in self.priors.keys():
                keys.append(key)
            #what if X_test = [1,5]
            for key in keys: 
                prior = self.priors[key]
                #prior = yes
                look_at = self.conditionals[key]
                #print(look_at)
                times = []
                for i in range(len(look_at)): #looking at each attribute
                    dict = look_at[i]
                    test_value = X_test[j][i]
                    if test_value in dict:
                        times.append(dict[test_value])
                    else:
                        times = []
                        break
                if times:
                    total = 1
                    for number in times:
                        total *= number

                    decision[key] = prior * total

            
            largest = 0 
            largest_pred = ""
            for key, val in decision.items():
                if val > largest:
                    largest = val
                    largest_pred = key


            predictions.append(largest_pred)

        return predictions



class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        options = {}
        for val in y_train:
            if val in options:
                options[val] += 1
            else:
                options[val] = 1
        # most frequent finding 
        self.most_common_label = max(options, key=options.get)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict()")
        
        return [self.most_common_label for x in X_test]

