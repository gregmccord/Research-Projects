----------------------------------------------
COS 424: Final Project (Spring 2018)
Author: Gregory McCord
NetID: gmccord
----------------------------------------------

The dataset used in this study can be found at: https://www.kaggle.com/Cornell-University/movie-dialog-corpus

The gmccord_FinalProject.zip file comprises the following file:

-data.py, called as follows
	python data.py [lr|nb|rf|lda|confusion]
[lr|nb|rf|lda|confusion] indicates the classifier to use:
lr = Logistic Regression
nb = Naive Bayes
rf = Random Forest
lda = latent dirichlet allocation
confusion = generate a confusion matrix for the genres

This is the master file for this project. When run, it will generate several files:
%[lr|nb|rf]%_log.txt - tracks the results of the methods
corpus.npy - the corpus for the lda method
movie_data.pkl - the movie data with appropriate labels
out_vocab.txt - the vocab list used in the movies
train_bow.csv - bow of words representation of the movies

It will also print a text to the terminal - the outputs of each model run and, in some cases, the influential
terms - and generate a plot of the ROC curves. All text output is the results of the models selected to
run. None of the generated files require manual inspection - model results, terms, and topic distributions are
all printed to the terminal.