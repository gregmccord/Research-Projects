# Author: Gregory McCord
# NetID: gmccord

from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse
import random

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


# Plot the fraction of viewers per movie and movies per viewer
def plotFraction(ratings,n_row,n_col):
    # Sum along column axis and sort
    count_nz_m = ratings.astype(bool).sum(axis=1)
    count_nz_l = list(count_nz_m.A1)
    count_nz = [float(i) / n_col for i in count_nz_l]
    count_nz.sort(reverse=True)
    
    # Plot 
    plt.plot(count_nz)
    plt.ylabel('Fraction of Viewers per Movie')
    plt.show()

    print(count_nz[2000]) # cutoff point

    # Sum along row axis and sort
    count_mv_m = ratings.astype(bool).sum(axis=0)
    count_mv_l = list(count_mv_m.A1)
    count_mv = [float(i) / n_row for i in count_mv_l]
    count_mv.sort(reverse=True)
    
    # Plot
    plt.plot(count_mv)
    plt.ylabel('Fraction of Movies per viewer')
    plt.show()

    print(count_mv[100000]) # cutoff point


# Cut the dimensionality of the matrix based on predefined values
def genData(ratings,movie_titles):
    # Sum along column axis
    (n_row, n_col) = ratings.shape
    count_nz_row = ratings.astype(bool).sum(axis=1)
    count_nz_row = list(count_nz_row.A1)
    count_nz = [float(i) / n_col for i in count_nz_row]

    # Remove rows without enough movies viewed
    count_mv_col = ratings.astype(bool).sum(axis=0)
    count_mv_col = list(count_mv_col.A1)
    count_mv = [float(i) / n_row for i in count_mv_col]
    cutoff_col = 0.0197147651007 # index 100000 of sorted count_nz
    mask_col = [i > cutoff_col for i in count_mv]

    indices_col = range(n_col)
    indices_col = [i for i in indices_col if mask_col[i]]

    ratings = ratings[:,indices_col]
    print(ratings.shape)

    count_nz_m = ratings.astype(bool).sum(axis=1)
    count_nz = list(count_nz_m.A1)
    count_nz.sort(reverse=True)
    
    plt.plot(count_nz)
    plt.ylabel('Fraction of Viewers per Movie')
    plt.show()

    # Remove columns
    cutoff_row = 5.39738736256e-05 # take top 2000 movies with genre data
    mask_row = [i > cutoff_row for i in count_nz]

    indices_row = range(n_row)
    indices_row = [i for i in indices_row if mask_row[i]]
    movies = movie_titles.iloc[indices_row,:]
    movies.to_csv('movies_small.csv',index=False,header=True)

    # Generate output
    df = ratings[indices_row,:]
    df2 = convertBinary(df)
    print(df2.shape)
    np.save('netflix_small.npy',df2)


# Convert the sparse matrix to dense with values -1 (missing), 0 (dislike), and 1 (like)
def convertBinary(ratings_sp):
    ratings = ratings_sp.todense()
    ratings[ratings == 0] = -1
    ratings[(ratings <= 2) & (ratings >= 0)] = 0
    ratings[ratings > 2] = 1

    return ratings


# Remove columns that don't have multiple labels in them (i.e. ratings are either all
# likes or all dislikes)
def multiLabel(ratings):
    b = [np.any(i == 0 for i in col) and np.any(i == 1 for i in col) for col in ratings.T]
    print(b)
    mask = np.array(b)
    print(all(isinstance(i, bool) for i in b))
    print(type(mask.dtype))
    df = ratings[:,mask]
    return df


def main():
    # This file consists of titles, release years, and genres associated with each ID
    movie_titles = pd.read_csv('movie_titles_genre.csv')

    # Load sparse data (0 = not viewed/rated)
    ratings = scipy.sparse.load_npz('netflix_full_csr.npz')
    (n_row, n_col) = ratings.shape
    print(ratings.shape)

    # Remove movies with no genre data
    mask_gen = [pd.isnull(row['G1']) for index,row in movie_titles.iterrows()]
    indices_gen = range(len(movie_titles))
    indices_gen = [i for i in indices_gen if mask_gen[i]]
    ratings = ratings[indices_gen,:]
    (n_row,n_col) = ratings.shape
    print(ratings.shape)

    plotFraction(ratings,n_row,n_col)
    genData(ratings,movie_titles)




if __name__ == "__main__":
    main()