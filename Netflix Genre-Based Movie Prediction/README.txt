-------------------------------------------------
COS 424: Netflix (Spring 2018)
Author: Gregory McCord
NetID: gmccord
-------------------------------------------------

The gmccord_hw3.zip file comprises the following files:

-imdb_genre.py, called as follows
    python3 imdb_genre.py
This program pulls genre data from imdb using the movies from movie_titles.txt and saves them in
'movie_titles_genre_%d.txt' % (end) where end is the end index. You must manually set the end
index number so that the TCP connection doesn't time out. It then combines all subsets
of the data and outputs 'movie_titles_genre.txt'.

-data_cleaning.py, called as follows
    python data_cleaning.py
This program reads movie genre data from 'movie_titles_genre.txt' and ratings data from
'netflix_full_csr.npz'. It then plots graphs of the fractions of viewers per movie and movies
per viewer, allowing you to shoose the threshold values for eliminating rows and columns. It
will then output the truncated movie list in 'movies_small.csv' and binary ratings in 'netflix_small.npy'.

-analysis.py, called as follows
    python analysis.py
This program will perform basic logistic regression on each user and print an out-of-sample
misclassification rate. It will also perform coordinate descent alternating between 
logistic regression and maximizing the log-liklihood of the latent variables. It will print
an out-of-sample misclassification rate and log-likelihood value at each iteration.
The misclassification rate is over all total samples