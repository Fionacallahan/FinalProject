import myutils


class MyRandomForestClassifier:
    """
    Represents a Random Forest classifier 

    Attributes: 


    Notes: 


    """
    def __init__(self):
        """
        Docstring for __init__
        
        :param self: Description
        """
        pass

    def fit(self):
        pass

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        conditionals(nested dict where each class goes to lsit of dicts): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
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

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
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

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
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

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
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

