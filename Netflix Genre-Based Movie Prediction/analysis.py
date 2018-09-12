# Author: Gregory McCord
# NetID: gmccord

import sys
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse
import random
from itertools import compress

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# The sigmoid function
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


# Randomly select values from each column to withhold from the training
def holdOut(ratings,n_row,n_col):
    random.seed(0)
    
    # Iterate over all non-missing data points
    (rows,cols) = np.where(ratings >= 0)
    cr = zip(cols,rows)
    cr.sort()

    i = 0
    nums = [0]
    for c,r in cr:
        if c == i:
            nums[c] += 1
        else:
            nums.append(1)
            i += 1
    
    # Randomly select points, save their value, and edit the matrix
    withheld = {}
    cumsum = 0
    for i,k in enumerate(nums):
        hold = int(k * 0.01) if int(k * 0.01) > 0 else 1
        withheld[i] = {}

        for j in range(hold):
            r = random.randint(0,k-1)
            if r != -1:
                (c,r) = cr[cumsum + r]
                withheld[c][r] = ratings[r,c]
                ratings[r,c] = -1

        cumsum += k

    # Determine the number of points withheld
    num_wh = 0
    for key in withheld:
        num_wh += len(withheld[key])
    print(num_wh)
    return withheld


# Remove users that have either all likes or all dislikes
def multiLabel(ratings):
    (n_row,n_col) = ratings.shape

    b = [any(i == 0 for i in col) and any(i == 1 for i in col) for col in ratings.T]
    print(np.sum(b))
    mask = np.array(b)
    df = ratings[:,mask]
    return df


# Baseline implementation - perform logistic regression on each person
def baseline(movie_genres,ratings,withheld):
    j=-1
    tot_class = []
    for col in ratings.T:
        j+= 1
        
        indices = (col >= 0)
        
        y = list(compress(col, indices))
        X = movie_genres[indices]
        

        clf = LogisticRegression(C=0.2).fit(X, y) # No regularization
    
        indices2 = (col < 0)

        pred = clf.predict(movie_genres)

        # Calculate out of sample misclassification rate
        comp = [withheld[j][i] == pred[i] for i,k in enumerate(indices2) if withheld[j].get(i) != None]
        num_class = np.sum(comp)

        tot_class.append(num_class)
        
    num_wh = 0
    for j in withheld:
        num_wh += len(withheld[j])

    avg_mis = 1 - np.sum(tot_class) / float(num_wh)
    print(avg_mis)


# Hold y_i fixed - perform logistic regression on each user
def y_fix(movie_genres,ratings,withheld,y):
    j=-1
    tot_class = []
    params = {}
    for col in ratings.T:
        j+= 1
        
        indices = (col >= 0)
        lab = list(compress(col, indices))

        X = movie_genres[indices].values
        y_new = np.array(y[indices])
        X = np.append(X,y_new,axis=1)

        clf = LogisticRegression(C=0.2).fit(X, lab) # No regularization
    
        indices2 = (col < 0)
        X2 = movie_genres[:].values
        X2 = np.append(X2,y,axis=1)

        pred = clf.predict(X2)

        # Calculate out of sample misclassification rate
        comp = [withheld[j][i] == pred[i] for i,k in enumerate(indices2) if withheld[j].get(i) != None]
        num_class = np.sum(comp)
        
        pars = clf.coef_.tolist()[0]
        params[j] = (pars[:-2],pars[-2:],float(clf.intercept_))
        tot_class.append(num_class)
    
    num_wh = 0
    for j in withheld:
        num_wh += len(withheld[j])

    avg_mis = 1 - np.sum(tot_class) / float(num_wh)
    return (avg_mis,params)


# Hold B_j,a_j fixed - optimize log-likelihood
def ba_fix(movie_genres,ratings,params):
    mat = movie_genres.values
    (n_row,n_col) = movie_genres.shape
    lam = 5

    y = []
    log_like = 0
    for i,row in enumerate(mat):

        best_y1 = 0
        best_y2 = 0
        max_ll = -1 * np.inf

        # Grid search for y_i
        for y_1 in np.arange(-2,2.1,0.5):
            for y_2 in np.arange(-2,2.1,0.5):
                log_like_i = 0

                # Compute log-likelihood
                for j in range(n_col):
                    rating = ratings[i,j]
                    if rating >= 0:
                        (beta,alpha,intercept) = params[j]
                        bx = np.dot(row,beta)
                        ay = np.dot(alpha,[y_1,y_2])
                        p = sigmoid(bx + intercept + ay)

                        term = p if rating == 1 else (1-p)
                        log_like_i += np.log(term)

                log_like_i -= lam * abs(y_1) + lam * abs(y_2)

                # Take max for this parameter set
                if log_like_i > max_ll:
                    max_ll = log_like_i
                    best_y1 = y_1
                    best_y2 = y_2

        log_like += max_ll
        y.append([best_y1,best_y2])

    return (log_like, np.array(y))


# Coordinate descent - iterate between logisitic regression and maximizing log-likelihood
def cd(movie_genres,ratings,withheld):
    abs_chg = 1000
    prev_ll = 0
    track_ll = []
    y = np.random.normal(0,1,[len(movie_genres),2]) # Initialization

    # Iteration
    while abs_chg > 0.5:
        (avg_mis,params) = y_fix(movie_genres,ratings,withheld,y)
        print(avg_mis)
        (log_like,y) = ba_fix(movie_genres,ratings,params)
        track_ll.append(log_like)
        print(log_like)

        abs_chg = abs(log_like - prev_ll)
        prev_ll = log_like

    # Plot time series of log-likelihood
    plt.plot(track_ll)
    plt.ylabel('Log-Likelihood per iteration')
    plt.show()


def main():
    # This file consists of titles, release years, and genres associated with each ID
    movie_titles = pd.read_csv('movies_small.csv')
    movie_titles.drop(['G1','G2','G3'],axis=1,inplace=True)
    movie_genres = movie_titles.drop(['ID','Year','Name'],axis=1)
    movie_titles = movie_titles[['ID','Year','Name']]

    # Load sparse data (0 = not viewed/rated)
    ratings = np.load('netflix_small.npy')
    (n_row, n_col) = ratings.shape
    print(ratings.shape)

    ratings = multiLabel(ratings)
    print(ratings.shape)
    np.save('netflix_small.npy',ratings)

    withheld = holdOut(ratings,n_row, n_col)
    baseline(movie_genres,ratings,withheld)
    #cd(movie_genres,ratings,withheld)
    


if __name__ == "__main__":
    main()