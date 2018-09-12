# Author: Gregory McCord
# NetID: gmccord

from pycorenlp import StanfordCoreNLP
import string
import csv
import re
from random import randint
import numpy as np

from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = np.genfromtxt(myfile,delimiter=',')
  return bagofwords

# Get the sentiment score
def get_score(text):

    # Connect to the server
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = text.lower()
    res = nlp.annotate(text,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 10000,
                       })
    
    # Average sentiment over sentences
    sum = 0
    tot_words = 0;
    for s in res["sentences"]:
        value = int(s["sentimentValue"]) - 2 # so that neutral is 0
        scaled_val = value * len(s["tokens"])
        tot_words += len(s["tokens"])
        sum += scaled_val
    score = sum / tot_words

    # If the review is "neutral", either randomly assign it as either positive or negative
    # sentiment or ignore the review all together
    if score == 0:
        #sent_score = randint(0,1)
        sent_score = -1

    if score < 0:
        sent_score = 0
    elif score > 0:
        sent_score = 1

    return sent_score

def main():
    pred = []
    printable = set(string.printable)

    y = read_bagofwords_dat('test_classes_0.txt')

    with open("test.txt", 'r') as f:
        lines = f.readlines()

    # Ignore any character that isn't readable (non-ascii) or is a number
    for line in lines:
        line = ''.join(filter(lambda x: x in printable, line))
        line = re.sub('[0-9]', ' ', line).strip()

        pred.append(get_score(line))

    # If ignoring "neutral" reviews, remove the correct labels from pred and y
    result = [(a,b) for (a,b) in zip(pred, y) if a != -1]
    pred = [r[0] for r in result]
    y = [r[1] for r in result]

    # Calculate misclassification rate
    mc_rate = zero_one_loss(y, pred)
    print(mc_rate)

    # Calculate the ROC curve
    roc_score = roc_auc_score(y, pred)
    print(roc_score)

    with open('stanford_scores.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(pred)

if __name__ == "__main__":
    main()