# Author: Gregory McCord
# NetID: gmccord

import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
import codecs
import csv
import sys

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from gensim.models.wrappers import LdaMallet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score


# Parse movie text and remove tabs from the script
def regex():
    with open('./movie-dialog-corpus/movie_lines_clean.tsv', 'w') as f_out:
        with open('./movie-dialog-corpus/movie_lines.tsv', 'r') as f:
            for line in f:
                re_output = re.search('(L[0-9]+)\t(u[0-9]+)\t(m[0-9]+)\t([^\t]+)\t(.+)',line)

                line_num = re_output.group(1)
                char_num = re_output.group(2)
                mov_num = re_output.group(3)
                char_name = re_output.group(4)
                text = re_output.group(5).replace('\t',' ')  

                f_out.write('%s\t%s\t%s\t%s\t%s\n' % (line_num,char_num,mov_num,char_name,text))


# Generate the movie script by combining all lines from each movie
def gen_movie_script(movies):
    script = {}
    scripts = []
    with open('./movie-dialog-corpus/movie_lines_clean.tsv', 'r') as f:
        index = -1
        for line in f:
            line = line.replace('\r','')
            line = line.strip('\n')
            re_output = re.search('(L[0-9]+)\t(u[0-9]+)\t(m[0-9]+)\t([^\t]+)\t(.*)',line)

            line_num = re_output.group(1)
            char_num = re_output.group(2)
            mov_num = re_output.group(3)
            char_name = re_output.group(4)
            text = re_output.group(5)

            # concatenate scripts or generate a new one
            if mov_num in script:
                script[mov_num] = text + '\n  ' + script[mov_num]
                scripts[index] = script[mov_num]
            else:
                script[mov_num] = text
                index += 1
                scripts.append(script[mov_num])

    # Generate a dataframe with the information
    script_df = pd.DataFrame(script.items(), columns=['movieID', 'script'])
    movie_data = pd.merge(movies, script_df, how='left',on='movieID')
    movie_data.to_pickle('movie_data.pkl')

    return (movie_data, scripts)


# Convert the scripts into a bow representation of the data (adapted from assignment 1)
def gen_bow(movie_data,scripts):
    chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
            '?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    docs = []
    words = {}

    for line in scripts:
        raw = line.decode('latin1')
        # remove noisy characters; tokenize
        raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
        tokens = word_tokenize(raw)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopWords]
        tokens = [wnl.lemmatize(t) for t in tokens]
        tokens = [porter.stem(t) for t in tokens]
        for t in tokens: 
            try:
                words[t] = words[t]+1
            except:
                words[t] = 1
        docs.append(tokens)

    # Generate vocab
    vocab = wordcount_filter(words)
    bow = find_wordcounts(docs, vocab)

    outfile = codecs.open('out_vocab.txt', 'w',"utf-8-sig")
    outfile.write("\n".join(vocab))
    outfile.close()

    return bow


# Obtain the counts for each word and remove it if <= 5 instances
def wordcount_filter(words, num=5):
   keepset = []
   for k in words.keys():
       if(words[k] > num):
           keepset.append(k)
   print "Vocab length:", len(keepset)
   return(sorted(set(keepset)))


# Obtain the word counts for each script - output the bow matrix
def find_wordcounts(docs, vocab):
   bagofwords = np.zeros(shape=(len(docs),len(vocab)), dtype=np.uint8)
   vocabIndex={}
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)


# Prepare the data to be used
def prep_data():
    # clean the data
    regex()

    # read the data in and produce the bow
    movies = pd.read_csv('./movie-dialog-corpus/movie_titles_one_hot.tsv', sep='\t')

    (movie_data,scripts) = gen_movie_script(movies)
    bow = gen_bow(movie_data,scripts)

    with open("train_bow.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(bow)


# Is the movie new or old
def isNew(row,low_year,high_year):
    if row['year'] <= low_year:
        return 0
    elif row['year'] >= high_year:
        return 1
    else:
        return -1
    

# Remove movies which are neither old nor new (set by low_year and high_year)
def check_years(movie_data,bow):
    low_year = 1990
    high_year = 2000
    print('low_year: %d high_year: %d' % (low_year,high_year))

    # Count the frequencies of each era
    years = np.array(movie_data['year'].tolist())
    freq = {}
    for year in years:
        if year in freq:
            freq[year] += 1
        else:
            freq[year] = 1

    # Keep only old or new movies
    classes = ((years >= high_year) | (years <= low_year))
    print('num_low: %d num_high: %d' % (np.sum((years <= low_year)),np.sum((years >= high_year))))
    df = movie_data[classes]
    df['year_label'] = df.apply(lambda row: isNew(row,low_year,high_year),axis=1)

    # normalize the counts
    sums = np.sum(bow,axis=1,dtype=np.float32)
    bow = bow / sums[:,None]

    X = bow[classes,:]

    return (X,df)


# Perform classification using Logistic Regression using the l2 penalty (adapted from assignment 1)
def lr(X_train,y_train,X_test,y_test):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxC = 0
    for c in np.arange(0.1,1,0.1):
        clf = LogisticRegression(penalty="l2", C=c).fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (c, mean))
        if mean > maxScore:
            maxScore = mean
            maxC = c

    # Train the model
    print("MaxC: %f" % maxC)
    print("MaxScore: %f" % maxScore)
    clf = LogisticRegression(penalty="l2", C=maxC).fit(X_train, y_train)

    # Predict labels for the test data
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)

    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y_test, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y_test, prob)
    print("ROC score: %f" % roc_score)

    return (mc_rate,roc_score)


# Perform classification using Multinomial Naive Bayes (adapted from assignment 1)
def nb(X_train,y_train,X_test,y_test):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxA = 0
    for a in np.arange(0.1,1,0.1):
        clf = MultinomialNB(alpha=a).fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        mean = np.mean(scores)
        print("A: %f and Score: %f" % (a, mean))
        if mean > maxScore:
            maxScore = mean
            maxA = a

    # Train the model
    print("MaxA: %f" % maxA)
    print("MaxScore: %f" % maxScore)
    clf = MultinomialNB(alpha=maxA).fit(X_train, y_train)

    # Predict labels for the test data
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    
    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y_test, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y_test, prob)
    print("ROC score: %f" % roc_score)

    return (mc_rate,roc_score)


# Perform classification using Random Forest using the gini criterion (adapted from assignment 1)
def rf(X_train,y_train,X_test,y_test):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxN = 0
    for n in np.arange(100,600,100):
        clf = RandomForestClassifier(n_estimators=n).fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        mean = np.mean(scores)
        print("N: %f and Score: %f" % (n, mean))
        if mean > maxScore:
            maxScore = mean
            maxN = n

    # Train the model
    print("MaxN: %f" % maxN)
    print("MaxScore: %f" % maxScore)
    clf = RandomForestClassifier(n_estimators=maxN).fit(X_train, y_train)

    # Predict labels for the test data
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)

    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y_test, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y_test, prob)
    curve = roc_curve(y_test, prob)
    print("ROC score: %f" % roc_score)

    return (mc_rate,roc_score,curve)


# Iterate over year and genres for the purpose of logistic regression
def lr_help(X_train,df_train,X_test,df_test,genres):
    with open('lr_log.txt','w') as f:
        # Choose response variable
        y_train = df_train['year_label']
        y_test = df_test['year_label']

        num_pos = np.sum(y_train) + np.sum(y_test)

        print('year')
        (mc_rate,roc_score) = lr(X_train,y_train,X_test,y_test)
        f.write("year\t=> MC rate: %f; ROC score: %f\n" % (mc_rate,roc_score))
        print('')

        # Iterate over genres
        for genre in genres:
            if genre not in ['animation','documentary','family','musical','short']:
                print(genre)

                # Choose response variable
                y_train = df_train[genre]
                y_test = df_test[genre]

                num_pos = np.sum(y_train) + np.sum(y_test)

                (mc_rate,roc_score) = lr(X_train,y_train,X_test,y_test)
                f.write("%s\t=> MC rate: %f; ROC score: %f; num_pos: %d\n" % (genre,mc_rate,roc_score,num_pos))
                print('')


# Iterate over year and genres for the purpose of naive bayes
def nb_help(X_train,df_train,X_test,df_test,genres):
    with open('nb_log.txt','w') as f:
        # Choose response variable
        y_train = df_train['year_label']
        y_test = df_test['year_label']

        num_pos = np.sum(y_train) + np.sum(y_test)

        print('year')
        (mc_rate,roc_score) = nb(X_train,y_train,X_test,y_test)
        f.write("year\t=> MC rate: %f; ROC score: %f\n" % (mc_rate,roc_score))
        print('')

        # Iterate over genres
        for genre in genres:
            if genre not in ['animation','documentary','family','musical','short']:
                print(genre)

                # Choose response variable
                y_train = df_train[genre]
                y_test = df_test[genre]

                num_pos = np.sum(y_train) + np.sum(y_test)

                (mc_rate,roc_score) = nb(X_train,y_train,X_test,y_test)
                f.write("%s\t=> MC rate: %f; ROC score: %f; num_pos: %d\n" % (genre,mc_rate,roc_score,num_pos))
                print('')


# Iterate over year and genres for the purpose of random forest
# Also plot ROC curves and obtain influential terms
def rf_help(X_train,df_train,X_test,df_test,genres,vocab):
    with open('rf_log.txt','w') as f:
        # Choose response variable
        y_train = df_train['year_label']
        y_test = df_test['year_label']

        num_pos = np.sum(y_train) + np.sum(y_test)

        print('year')
        roc_curves = {}

        # Get influential terms
        get_importance(X_train,y_train,X_test,y_test,vocab)
        (mc_rate,roc_score,roc_curve) = rf(X_train,y_train,X_test,y_test)
        roc_curves['year'] = roc_curve
        f.write("year\t=> MC rate: %f; ROC score: %f\n" % (mc_rate,roc_score))
        print('')

        
        for genre in genres:
            if genre not in ['animation','documentary','family','musical','short']:
                print(genre)
                # Choose response variable
                y_train = df_train[genre]
                y_test = df_test[genre]

                num_pos = np.sum(y_train) + np.sum(y_test)

                # Get influential terms
                get_importance(X_train,y_train,X_test,y_test,vocab)
                (mc_rate,roc_score,roc_curve) = rf(X_train,y_train,X_test,y_test)
                roc_curves[genre] = roc_curve
                f.write("%s\t=> MC rate: %f; ROC score: %f; num_pos: %d\n" % (genre,mc_rate,roc_score,num_pos))
                print('')

        # Plot ROC curves
        plt.figure()
        for x in roc_curves:
            if x in ['adventure','action','drama','sci-fi','crime','year']:
                plt.plot(roc_curves[x][0], roc_curves[x][1],lw=2, label='%s' % x)
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()


# Print out the top 20 influential terms for the given response labels
def get_importance(X_train,y_train,X_test,y_test,vocab):
    # Tune the hyperparameter
    maxScore = float("-inf")
    maxN = 0
    for n in np.arange(100,600,100):
        clf = RandomForestClassifier(n_estimators=n).fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        mean = np.mean(scores)
        print("N: %f and Score: %f" % (n, mean))
        if mean > maxScore:
            maxScore = mean
            maxN = n

    # Train the model
    print("MaxN: %f" % maxN)
    print("MaxScore: %f" % maxScore)
    clf = RandomForestClassifier(n_estimators=maxN).fit(X_train, y_train)

    # get importances in sorted order
    importances = clf.feature_importances_
    index = range(0,len(importances))
    joint = zip(importances, index)
    joint.sort(key = lambda t: t[0], reverse=True)

    # Print top vocab terms
    for x in range(0,20):
        (a,b) = joint[x]
        print('%s' % vocab[b])


# Convert the bow into a sparse representation for MALLET
def text_to_corpus(bow):
    corpus = []
    for row in bow:
        doc = []
        for i in range(len(row)):
            if row[i] > 0:
                doc.append((i,row[i]))
        corpus.append(doc)

    np.save('corpus.npy',corpus)
    return corpus


# Perform LDA (Latent Dirichlet Allocation) using the gensim MALLET wrapper (version 2.0.8)
# download from: http://mallet.cs.umass.edu/download.php and place in the same folder as the file
def lda(bow,df,vocab):
    # Generate and load corpus
    corpus = text_to_corpus(bow)
    corpus = np.load('corpus.npy')

    path_to_mallet = './mallet-2.0.8/bin/mallet'
    model = LdaMallet(path_to_mallet, corpus=corpus, num_topics=5, workers=4,id2word=vocab)
    res = model.print_topics(num_topics=-1, num_words=50)

    # print response
    for x in res:
        print(x)
    for x in model[corpus]:
        print(x)


# Generate a confusion matrix for the genre classification models
def confusion_mat(X,df,X_train,df_train,genres):
    clfs = {}
    print('Train')
    for genre in genres:

        # Only include the top 10 genres
        if genre in ['action','adventure','comedy','crime','drama','horror','mystery',
                    'romance','sci-fi','thriller']:
            print(genre)
            y_train = df_train[genre]

            # Tune hyperparameter and run models
            maxScore = float("-inf")
            maxN = 0
            for n in np.arange(100,600,100):
                clf = RandomForestClassifier(n_estimators=n).fit(X_train, y_train)
                scores = cross_val_score(clf, X_train, y_train, cv=5)
                mean = np.mean(scores)
                print("N: %f and Score: %f" % (n, mean))
                if mean > maxScore:
                    maxScore = mean
                    maxN = n

            # Train the model
            print("MaxN: %f" % maxN)
            print("MaxScore: %f" % maxScore)
            clfs[genre] = RandomForestClassifier(n_estimators=maxN).fit(X_train, y_train)
    
    # Test the models
    print('Test')
    res = []
    for genre in genres:
        if genre in ['action','adventure','comedy','crime','drama','horror','mystery',
                    'romance','sci-fi','thriller']:
            
            # For each set of labels...
            print(genre)
            mask = (df[genre] == 1)
            test_set = X[mask,]

            # Iterate over genres and test the labels using each model
            res_inner = []
            for x in genres:
                if x in ['action','adventure','comedy','crime','drama','horror','mystery',
                        'romance','sci-fi','thriller']:
                    print(x)
                    pred = clfs[x].predict(test_set)
                    c_rate = sum(pred) / float(len(pred))
                    res_inner.append(c_rate)
            res.append(res_inner)

    # print results
    print(genres)
    for x in res:
        print(x)


def main(argv):
    if (len(argv) != 1):
        print 'Usage: \n python data.py [lr|nb|rf|lda|confusion]'
        sys.exit(2)

    model = argv[0]

    # Prep and clean the data
    prep_data()
    movie_data = pd.read_pickle('movie_data.pkl')
    bow = np.genfromtxt('train_bow.csv', delimiter=',')
    genres = list(movie_data.columns[5:-1])

    # Generate a list of the words used
    vocab = {}
    with open('out_vocab.txt','r') as f_in:
        index = 0
        for line in f_in:
            vocab[index] = line.strip()
            index += 1

    # Remove movies that are neither old nor new
    (X,df) = check_years(movie_data,bow)
    
    # Split into training and testing sets
    np.random.seed(1)
    (X_train, X_test, df_train, df_test) = train_test_split(X, df, test_size=0.3)
    
    if model == 'lr':
        lr_help(X_train,df_train,X_test,df_test,genres)
    elif model == 'nb':
        nb_help(X_train,df_train,X_test,df_test,genres)
    elif model == 'rf':
        rf_help(X_train,df_train,X_test,df_test,genres,vocab)
    elif model == 'lda':
        lda(bow,df,vocab)
    elif model == 'confusion':
        confusion_mat(X,df,X_train,df_train,genres)
    else:
        print('Usage: \n python data.py [lr|nb|rf|lda|confusion]')
    

if __name__ == "__main__":
    main(sys.argv[1:])