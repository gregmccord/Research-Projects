# Author: Gregory McCord
# NetID: gmccord

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import cPickle as cp

# Perform Multinomial Logistic Regression
def lr(df,labels):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxC = 0
    bestCLF = None
    for c in np.arange(0.1,0.3,0.1):
        clf = LogisticRegression(C=c,penalty="l2",dual=False,
                                multi_class="multinomial",solver="saga").fit(df, labels)
        scores = cross_val_score(clf, df, labels, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (c, mean))
        if mean > maxScore:
            maxScore = mean
            maxC = c
            bestCLF = clf

    print("MaxC: %f" % maxC)
    print("MaxScore: %f" % maxScore)

    return bestCLF


# Perform classification using Support Vector Machine with l1 penalty and linear kernel
def svm(df,labels):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxC = 0
    bestCLF = None
    for c in np.arange(0.1,0.5,0.1):
        clf = LinearSVC(C=c, penalty="l1", dual=False).fit(df, labels)
        scores = cross_val_score(clf, df, labels, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (c, mean))
        if mean > maxScore:
            maxScore = mean
            maxC = c
            bestCLF = clf

    # Train the model
    print("MaxC: %f" % maxC)
    print("MaxScore: %f" % maxScore)

    return bestCLF


# Perform Multinomial Naive Bayes
def mnb(df,labels):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxA = 0
    bestCLF = None
    for a in np.arange(0.1,0.5,0.1):
        clf = MultinomialNB(alpha=a).fit(df, labels)
        scores = cross_val_score(clf, df, labels, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (a, mean))
        if mean > maxScore:
            maxScore = mean
            maxA = a
            bestCLF = clf

    print("MaxA: %f" % maxA)
    print("MaxScore: %f" % maxScore)

    return bestCLF


# Perform Gaussian Naive Bayes
def gnb(df,labels):

    clf = GaussianNB().fit(df, labels)
    scores = cross_val_score(clf, df, labels, cv=5)
    mean = np.mean(scores)
    print("Mean: %f" % mean)

    return clf


# Perform Random Forest Classifier
def rf(df,labels):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxN = 0
    bestCLF = None
    for n in np.arange(100,600,500):
        clf = RandomForestClassifier(n_estimators=n).fit(df, labels)
        scores = cross_val_score(clf, df, labels, cv=5)
        mean = np.mean(scores)
        print("N: %f and Score: %f" % (n, mean))
        if mean > maxScore:
            maxScore = mean
            maxN = n
            bestCLF = clf

    print("MaxN: %f" % maxN)
    print("MaxScore: %f" % maxScore)

    return bestCLF


def main(argv):
    # Ensure the input is correct
    if (len(argv) != 3):
        print 'Usage: \n python classificationDiscrete.py '
              '[eviction/layoff/jobTraining] '
              '[lr/mnb/gnb/svm/rf] '
              '[all/both/question/constructed]'
        sys.exit(2)

    goal = argv[0]
    model = argv[1]
    dataset = argv[2]

    # Read in data
    df = pd.read_csv("%s_data.csv" % (dataset), low_memory=False)
    df.dropna(subset=[goal])

    # Don't train on labels
    length = len(df.columns)
    pre_labels = df[goal]
    pre_data = df.drop(columns=['gpa','grit','materialHardship','eviction','layoff','jobTraining'])

    # Ensure all values are numbers
    data = np.nan_to_num(pre_data)
    labels = np.nan_to_num(pre_labels)

    # choose correct model
    if model == 'lr':
        clf = lr(data,labels)
    elif model == 'mnb':
        clf = mnb(data,labels)
    elif model == 'gnb':
        clf = gnb(data,labels)
    elif model == 'svm':
        clf = svm(data,labels)
    elif model == 'rf':
        clf = rf(data,labels)

    # output model
    with open('model_%s_%s_%s.pkl' % (model,goal,dataset), 'wb') as f:
        cp.dump(clf, f)

    

if __name__ == "__main__":
    main(sys.argv[1:])