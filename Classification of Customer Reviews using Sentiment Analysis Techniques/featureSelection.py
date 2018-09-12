# Author: Gregory McCord
# NetID: gmccord

import numpy as np
import csv
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = np.genfromtxt(myfile,delimiter=',')
  return bagofwords

def main():
    bow = read_bagofwords_dat('out_bag_of_words_5.csv')
    print(bow.shape)
    classes = read_bagofwords_dat('out_classes_5.txt')
    print(classes.shape)

    # Perform feature selection and cross validation to determine the optimal
    # number of features to select
    maxScore = float("-inf")
    maxC = 0
    for c in np.arange(0.1,0.5,0.01):
        lsvc = LinearSVC(C=c, penalty="l1", dual=False).fit(bow, classes)
        scores = cross_val_score(lsvc, bow, classes, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (c, mean))
        if mean > maxScore:
            maxScore = mean
            maxC = c

    # Having tuned the hyperparameter C, fit the model, and save the selected features
    # by writing to a new output file
    print("MaxC: %f" % maxC)
    print("MaxScore: %f" % maxScore)
    lsvc = LinearSVC(C=maxC, penalty="l1", dual=False).fit(bow, classes)
    model = SelectFromModel(lsvc, prefit=True)
    bow_new = model.transform(bow)
    print(bow_new.shape)
    with open("out_bag_of_words_transform"+".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(bow_new)

    # Save new vocabulary for use with test dataset
    with open("out_vocab_5.txt") as f:
        vocab = [line.rstrip('\n') for line in f]
    terms_kept = model.get_support()
    vocab_transform = [i for (i,b) in zip(vocab,terms_kept) if b == True]
    with open("out_vocab_transform"+".txt", "wb") as f:
        f.write("\n".join(vocab_transform))

if __name__ == "__main__":
    main()