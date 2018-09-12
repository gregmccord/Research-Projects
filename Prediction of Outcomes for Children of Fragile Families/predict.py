from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np
import cPickle as cp
import math

# The sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main(argv):
    # Ensure the input is correct
    if (len(argv) != 2):
        print 'Usage: \n python predict.py '
              '[lr/mnb/gnb/svm/rf] '
              '[all/both/question/constructed]'
        sys.exit(2)

    disModel = argv[0]
    dataset = argv[1]

    # Read in data
    df = pd.read_csv("%s_data_pre.csv" % (dataset), low_memory=False)
    data = np.nan_to_num(df)

    # Save results in a dictionary before converting to dataframe
    indices = range(1,len(df)+1)
    predictions = {'challengeID':indices}

    # Dummy data for continuous variables
    for goal in ['gpa', 'grit', 'materialHardship']:

        # Predict labels for the test data
        predictions[goal] = indices

    for goal in ['eviction','layoff','jobTraining']:
        # read model
        with open('model_%s_%s_%s.pkl' % (disModel,goal,dataset), 'rb') as f:
            clf = cp.load(f)

        # Predict labels for the test data
        if disModel == 'svm':
            # Predict Probabilities
            confidence = clf.decision_function(data)
            pred = map(sigmoid, confidence)
            predictions[goal] = pred

            # Predict Binary
            # pred = clf.predict(data)
            # predictions[goal] = pred
        else:
            # Predict Probabilities
            pred = clf.predict_proba(data)[:,1:]
            predictions[goal] = [sublist[0] for sublist in pred]

            # Predict Binary
            # pred = clf.predict(data)
            # predictions[goal] = pred

    # Output predictions
    output = pd.DataFrame(predictions)
    output = output[['challengeID','eviction','layoff','jobTraining']]
    output.to_csv("prediction.csv", index=False)    

if __name__ == "__main__":
    main(sys.argv[1:])