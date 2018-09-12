# Author: Gregory McCord
# NetID: gmccord

import pandas as pd
import numpy as np
import cPickle as cp

def main(argv):
    # Ensure the input is correct
    if (len(argv) != 1):
        print 'Usage: \n python topTerms.py '
              '[all/both/question/constructed]'
        sys.exit(2)

    dataset = argv[0]

    # Get variables
    df = pd.read_csv("metadata.csv", low_memory=False)
    features = df["new_name"].tolist()

    # Iterate over discrete response variables
    for goal in ['eviction','layoff','jobTraining']:
        print(goal)

        # read model
        with open('model_rf_%s_%s.pkl' % (goal,dataset), 'rb') as f:
            clf = cp.load(f)

        # get importances in sorted order
        importances = clf.feature_importances_
        index = range(0,len(importances))
        joint = zip(importances, index)
        joint.sort(key = lambda x: x[0], reverse=True)

        # Print top vocab terms
        for x in range(0,10):
            (a,b) = joint[x]
            print(features[b])

        print('\n')

if __name__ == "__main__":
    main(sys.argv[1:])