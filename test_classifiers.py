import numpy as np
from scipy import stats
import myutils

from myclassifiers import MyNaiveBayesClassifier,\
    MyDummyClassifier,\
    MyDecisionTreeClassifier

# implement test random forest 


def test_decision_tree_classifier_fit():

    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]

    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]


    # the building of the tree using entropy from deskcheck
    interview_tree_solution =   ["Attribute", "att0",                        
                                ["Value", "Junior",                     
                                    ["Attribute", "att3",               
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]      
                                        ],
                                        ["Value", "yes",
                                            ["Leaf", "False", 2, 5]
                                        ]
                                    ]
                                ],
                                ["Value", "Mid",
                                    ["Leaf", "True", 4, 14] 
                                ],
                                ["Value", "Senior",
                                    ["Attribute", "att2",
                                        ["Value", "no",
                                            ["Leaf", "False", 3, 5] 
                                        ],
                                        ["Value", "yes",
                                            ["Leaf", "True", 2, 5] 
                                        ]
                                    ] 
                                ]
                            ]
    

    myInterviewTree = MyDecisionTreeClassifier()
    myInterviewTree.fit(X_train_interview, y_train_interview)
    assert myInterviewTree.tree == interview_tree_solution
    

    header_phone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_phone = [[2, 3, "fair"],
                     [1, 3, "excellent"], 
                     [1, 3, "fair"], 
                     [2, 2, "fair"], 
                     [2, 1, "fair"], 
                     [2, 1, "excellent"],
                     [2, 1, "excellent"],
                     [1, 2, "fair"], 
                     [1, 1, "fair"],
                     [2, 2, "fair"],
                     [1, 2, "excellent"],
                     [2, 2, "excellent"], 
                     [2, 3, "fair"], 
                     [2, 2, "excellent"], 
                     [2, 3, "fair"]]
    y_train_phone = ["yes", "no", "no", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # building of tree using entropy from hand 
    phone_tree_soluti = ["Attribute", "att0", 
                            ["Value", 1,
                                ["Attribute", "att1",   
                                    ["Value", 1, ["Leaf", "yes", 1, 5]],
                                    ["Value", 2,
                                        ["Attribute", 'att2', 
                                            ['Value', 'excellent', 
                                                ["Leaf", "yes", 1, 2]],
                                            ['Value', 'fair', 
                                                ["Leaf", "no", 1, 2]]]
                                    ],
                                    ["Value", 3, ["Leaf", "no", 2, 5]]
                                ]
                            ],
                            ["Value", 2,
                                ["Attribute", "att2", # now att2 makes more sense 
                                    ["Value", "excellent",
                                        ['Attribute', 'att1', 
                                            ['Value', 1, 
                                                ['Leaf', 'no', 2, 4]],
                                            ['Value', 2, 
                                                ['Leaf', 'yes', 2, 4]],
                                            ['Value', 3, 
                                                ['Leaf', 'no', 0, 4]]]
                                    ],
                                    ["Value", "fair",
                                        ["Leaf", "yes", 6, 10] # clumps all remaining into one
                                    ]
                                ]
                            ]
                        ]
    print("entering phone tree classification")
    myPhoneTree = MyDecisionTreeClassifier()
    myPhoneTree.fit(X_train_phone, y_train_phone)
    assert myPhoneTree.tree == phone_tree_soluti

def test_decision_tree_classifier_predict():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]

    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    myInterviewTree = MyDecisionTreeClassifier()
    myInterviewTree.fit(X_train_interview, y_train_interview)
    # from interview dataset 
    X1 = ["Junior", "Java", "yes", "no"] 
    #SHOULD BE True
    X2 = ["Junior", "Java", "yes", "yes"] 
    #SHOULD BE false
    predicted = ["True", "False"]

    predictions = myInterviewTree.predict([X1, X2])
    assert predictions == predicted
    


    # now for the phone dataset 
    header_phone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_phone = [[2, 3, "fair"],
                     [1, 3, "excellent"], 
                     [1, 3, "fair"], 
                     [2, 2, "fair"], 
                     [2, 1, "fair"], 
                     [2, 1, "excellent"],
                     [2, 1, "excellent"],
                     [1, 2, "fair"], 
                     [1, 1, "fair"],
                     [2, 2, "fair"],
                     [1, 2, "excellent"],
                     [2, 2, "excellent"], 
                     [2, 3, "fair"], 
                     [2, 2, "excellent"], 
                     [2, 3, "fair"]]
    y_train_phone = ["yes", "no", "no", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    X3 = [2, 2, "fair"] 
    # SHOULD be yes 
    X4 = [1, 1, "excellent"] 
    # SHOULD be yes based on majority vote and tree structure 
    myPhoneTree = MyDecisionTreeClassifier()
    myPhoneTree.fit(X_train_phone, y_train_phone)
    predictions = myPhoneTree.predict([X3, X4])
    predicted = ["yes", "yes"]
    assert predicted == predictions





def test_naive_bayes_classifier_fit():
    # use 8 instance training set done in class 
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]    
        
    NaiveBayes1 = MyNaiveBayesClassifier()
    NaiveBayes1.fit(X_train_inclass_example, y_train_inclass_example)


    # assert against our desk check of priors and conditional probabilities 
    """
    P(result = yes) : 5/8
    P(result = no) : 3/8

    CONDITIONALS: 
    P(Att1 = 1 | yes) = 4/5
    P(Att1 = 2 | yes) = 1/5
    P(Att1 = 1 | no) = 2/3
    P(Att1 = 2 | no) = 1/3

    P(Att2 = 5 | yes) = 2/5
    P(Att2 = 6 | yes) = 3/5
    P(Att2 = 5 | no) = 2/3
    P(Att2 = 6 | no) = 1/3
    
    """
    assert(NaiveBayes1.priors ==  {"yes": 5/8, "no": 3/8})

    # somehow do conditionals 
    # basic structure: {class: [ {feature_value: count}, ... ] }
    conditional_dict_class = {"yes": [{1: 4/5, 2: 1/5}, {5: 2/5, 6: 3/5}], "no" : [{1: 2/3, 2: 1/3}, {5: 2/3, 6: 1/3}]}
    assert(NaiveBayes1.conditionals == conditional_dict_class)

    # 15 instance training set from LA7
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]


    NaiveBayes2 = MyNaiveBayesClassifier()
    NaiveBayes2.fit(X_train_iphone, y_train_iphone)
    #assert against priors and conditional probabilities 
    assert(NaiveBayes2.priors == {"yes": 10/15, "no": 5/15})

    #figure out: 
    conditionals = {"yes": [{1: 2/10, 2: 8/10}, {1: 3/10, 2: 4/10, 3: 3/10}, {"fair": 7/10, "excellent": 3/10}], 
                    "no": [{1: 3/5, 2: 2/5}, {1: 1/5, 2: 2/5, 3: 2/5}, {"fair": 2/5, "excellent": 3/5}]}
    assert(NaiveBayes2.conditionals == conditionals)

    # assert Bramer 3.2 train data set 
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]



    NaiveBayes3 = MyNaiveBayesClassifier()
    NaiveBayes3.fit(X_train_train, y_train_train)

    #assert against priors and probabilities 
    """
    Prior_on time: 14/20
    Prior_late = 2/20
    Prior_very late = 3/20
    Prior_cancelled = 1/20

    
    # just check three sets of conditionals 
    P(weekday | on time) = .64
    P(weekday | late) = .5
    P(weekday | very late) = 1
    P (weekday | cancelled = 0

    P(winter | on time) = .14
    P(winter | late) = 1
    P(winter | very late) = .67
    P (winter | cancelled = 0

    P(high wind | on time) = .29
    P(high wind | late) = .5
    P(high wind | very late) = .33
    P (high wind | cancelled = 1
    """
    #assert against priors and conditional probabilities 
    assert(NaiveBayes3.priors == {"on time": 14/20, "late": 2/20, "very late": 3/20, "cancelled": 1/20})

    #figure out:
    # NOTE: ANYTHING WITH ZERO IS OUT 
    conditionals = {"on time": [{"weekday": 9/14, "saturday": 2/14, "sunday": 1/14, "holiday": 2/14}, {"spring": 4/14, "summer": 6/14, "autumn": 2/14, "winter": 2/14}, {"none": 5/14, "high": 4/14, "normal": 5/14}, {"none": 5/14, "slight": 8/14, "heavy": 1/14}], 
                    "late": [{"weekday": 1/2, "saturday": 1/2}, {"winter": 1}, {"high": .5, "normal": .5}, {"none": .5, "heavy": .5}],
                    "very late": [{"weekday": 1}, {"autumn": 1/3, "winter": 2/3}, {"high": 1/3, "normal": 2/3}, {"none": 1/3, "heavy": 2/3}], 
                    "cancelled": [{"saturday": 1}, {"spring": 1}, {"high": 1}, {"heavy": 1}]}
    assert(NaiveBayes3.conditionals == conditionals)


def test_naive_bayes_classifier_predict():
    # first assertion: P should be yes 
    # P (result = yes | X) = .2
    # P(result = no | X) = .16
    # use 8 instance training set done in class 
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]    
        
    NaiveBayes1 = MyNaiveBayesClassifier()
    NaiveBayes1.fit(X_train_inclass_example, y_train_inclass_example)
    predictions = NaiveBayes1.predict([[1,5]])

    assert (predictions == ["yes"])

    # second assertion: YES
    # USE [standing = 2, job status = 2, credit rating = fair] 
    # P (buys_iphone = no | X) = .021
    # P (buys_iphone = yes | X) = .149

    # part b: 
    # USE [standing = 1, job_status = 1, credit_rating = excellent]
    # P (buys_iphone = no | X) = .024
    # P (buys_iphone = yes | X) = .012
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]


    NaiveBayes2 = MyNaiveBayesClassifier()
    NaiveBayes2.fit(X_train_iphone, y_train_iphone)
    predictions = NaiveBayes2.predict([[2,2,"fair"], [1,1,"excellent"]])
    assert(predictions==["yes", "no"])


    # third assertion: very late 
    # use ["weekday", "winter", "high", "heavy"]

    # assertion: on time 
    # use [weekday, summer, high, heavy ]

    # also use: [sunday, summer, normal, slight]
    # on time 
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]



    NaiveBayes3 = MyNaiveBayesClassifier()
    NaiveBayes3.fit(X_train_train, y_train_train)
    predictions = NaiveBayes3.predict([["weekday", "winter", "high", "heavy"], 
                        ["weekday", "summer", "high", "heavy"],
                        ["sunday", "summer", "normal", "slight"]])

    assert(predictions == ["very late", "on time", "on time"])





def test_dummy_classifier_fit():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train = [[1]] * len(y_train)

    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)

    assert dummy_clf.most_common_label == "yes"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_train = [[1]] * len(y_train)

    dummy_clf2 = MyDummyClassifier()
    pred2 = dummy_clf2.fit(X_train, y_train)

    assert dummy_clf2.most_common_label == "no"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.3, .5]))
    X_train = [[1]] * len(y_train)

    dummy_clf3 = MyDummyClassifier()
    pred3 = dummy_clf3.fit(X_train, y_train)

    assert dummy_clf3.most_common_label == "maybe"


def test_dummy_classifier_predict():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train = [[1]] * len(y_train)

    dummy_clf = MyDummyClassifier()
    pred = dummy_clf.fit(X_train, y_train)
    X_test = [[0,2]]
    classifier = dummy_clf.predict(X_test)

    assert classifier == ["yes"]

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_train = [[1]] * len(y_train)

    dummy_clf2 = MyDummyClassifier()
    pred2 = dummy_clf2.fit(X_train, y_train)
    X_test = [[0,2]]
    classifier2 = dummy_clf2.predict(X_test)

    assert classifier2 == ["no"]
    

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.3, .5]))
    X_train = [[1]] * len(y_train)

    dummy_clf3 = MyDummyClassifier()
    pred3 = dummy_clf3.fit(X_train, y_train)

    X_test = [[0,2]]
    classifier3 = dummy_clf3.predict(X_test)

    assert classifier3 == ["maybe"]

