# Author: Gregory McCord
# NetID: gmccord

from imdb import IMDb # This package is only available for Python 3
import numpy as np
import pandas as pd

def pull():
    end = 18000

    ia = IMDb()

    movie_titles = pd.read_csv('movie_titles.txt', header = None, names = ['ID','Year','Name'],encoding='latin-1')
    movie_titles[['Year']] = movie_titles[['Year']].astype('object')
    movie_titles = movie_titles[end-1000:end]
    genres = {}

    for index, movie in movie_titles.iterrows():
        movieList = ia.search_movie(movie['Name'])
        genre = [movie['ID']]
        print(movie['ID']) # View progress of method

        if not movieList:
            genre.extend(['','',''])
        else:
            record = ia.get_movie(movieList[0].movieID)
            try:
                genre_np = np.array(record['genre'])
                genre_np.resize((3))
                genre.extend(list(genre_np))
            except:
                genre.extend(['','',''])

        genres[movie['ID']] = genre

    df = pd.DataFrame.from_dict(genres,orient='index')
    df.columns=['ID','G1','G2','G3']
    movie_data = pd.merge(movie_titles, df, how='left',on='ID')
    movie_data.to_csv('movie_titles_genre_%d.txt' % (end),index=False,header=False)


def assemble():
    with open('movie_titles_genre.txt','w',encoding='latin-1') as f:
        for i in [1000 * v for v in range(1,19)]:
            with open('movie_titles_genre_%d.txt' % (i), 'r',encoding='latin-1') as myfile:
                data = myfile.read()
                print(data, file=f,end='')


def main():
    #pull()
    assemble()
    

if __name__ == "__main__":
    main()