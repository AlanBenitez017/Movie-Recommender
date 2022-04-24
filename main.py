import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import io
import csv


  
creditsDF = pd.read_csv('tmdb_5000_credits.csv', index_col=0)
moviesDF = pd.read_csv('tmdb_5000_movies.csv', index_col=0)


print(moviesDF.head())
print(creditsDF.head())

moviesDF = pd.merge(creditsDF, moviesDF, on = "id", how = "inner")


print(moviesDF.head())

C = moviesDF["vote_average"].mean()
m = moviesDF["vote_count"].quantile(0.9)
print("C: ", C)
print("m: ", m)

newMoviesDF = moviesDF.copy().loc[moviesDF["vote_count"] >= m]
print(newMoviesDF.shape)

def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)

newMoviesDF["score"] = newMoviesDF.apply(weighted_rating, axis=1)
newMoviesDF = newMoviesDF.sort_values('score', ascending=False)

newMoviesDF[["title_y", "vote_count", "vote_average", "score"]].head(5)

def plot():
    popularity = moviesDF.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title_y"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 movies")
    plt.xlabel("Popularity")
    plt.show()

plot()

print(moviesDF["overview"].head(5))

tfidf = TfidfVectorizer(stop_words="english")
moviesDF["overview"] = moviesDF["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(moviesDF["overview"])
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

indices = pd.Series(moviesDF.index, index=moviesDF["title_y"]).drop_duplicates()
print(indices.head())

def get_recommendations(title, cosine_sim=cosine_sim):
    """
    in this function,
        we take the cosine score of given movie
        sort them based on cosine score (movie_id, cosine_score)
        take the next 10 values because the first entry is itself
        get those movie indices
        map those indices to titles
        return title list
    """
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    # (a, b) where a is id of movie, b is sim_score

    movies_indices = [ind[0] for ind in sim_scores]
    movies = moviesDF["title_y"].iloc[movies_indices]
    return movies

print("Content based filtering")
val = input("Enter the name of the movie that you would like to see the Recomendations: ")
print("Recommendations for " + val)
print(get_recommendations(val))



