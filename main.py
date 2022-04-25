# ----------------------------------------------------------------------------
# Student Names: Alan Benitez and Justin Gilmer
# File Name: main.py
# Project 2 - COP 4601
# 24 April 2022
# Movie Recommender using content-based filtering. The program attempts to guess what a user may like based on a title chosen by the user
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import sys

class MovieRecommender:

    # Overloaded constructor 
    def __init__(self, filename1, filename2, creditsDF, moviesDF):
        self.filename1 = filename1
        self.filename2 = filename2
        self.creditsDF = creditsDF
        self.moviesDF = moviesDF

    # Method to read the datasets
    def readDatasets(self, filename1, filename2):
        self.creditsDF = pd.read_csv(filename1, index_col=0)
        self.moviesDF = pd.read_csv(filename2, index_col=0)
        print(self.moviesDF.head())
        print(self.creditsDF.head())
        self.moviesDF = pd.merge(self.creditsDF, self.moviesDF, on = "id", how = "inner")

    # Method to calculate the average and the quantiles
    def calculateAverageAndQuantiles(self, moviesDF):
        averageVote = moviesDF["vote_average"].mean()
        quantile90percent = moviesDF["vote_count"].quantile(0.9)
        print("Average Vote: ", averageVote)
        print("Quantile 90%: ", quantile90percent)

        moviesDF = moviesDF.copy().loc[moviesDF["vote_count"] >= quantile90percent]
        print(moviesDF.shape)
        self.updateAndPrint(moviesDF, averageVote, quantile90percent)
        
    # Helper method to weight the rating of every movie
    def weighted_rating(self, df, averageVote, quantile90percent):
        voteCount = df["vote_count"]
        R = df["vote_average"]
        #print("R = " + R)
        return (voteCount/(voteCount + quantile90percent) * R) + (quantile90percent/(voteCount + quantile90percent) * averageVote)
    
    # Adds a new column named score and populates it with the weighted rating of every row, then sorts it by scores and prints the first 5
    def updateAndPrint(self, moviesDF, averageVote, quantile90percent):
        try:
            moviesDF["score"] = self.moviesDF.apply(self.weighted_rating(moviesDF, averageVote, quantile90percent), axis=1)   #applying the weighted_rating for each row
            moviesDF = moviesDF.sort_values('score', ascending=False)
        except AssertionError as msg:
            print(msg)
        moviesDF.head(5)

    # Plot a graph of the top 10 movies based on popularity of the movies
    def plot(self, moviesDF):
        popularity = moviesDF.sort_values("popularity", ascending=False)
        plt.figure(figsize=(12, 6))
        plt.barh(popularity["title_y"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
        plt.gca().invert_yaxis()
        plt.title("Top 10 movies")
        plt.xlabel("Popularity")
        plt.show()

    # Does the cosine similarity using the overview column of the dataset
    def cosineSimilarity(self, moviesDF):
        print(moviesDF["overview"].head(5))

        tfidf = TfidfVectorizer(stop_words="english")
        moviesDF["overview"] = moviesDF["overview"].fillna("")

        tfidf_matrix = tfidf.fit_transform(moviesDF["overview"])
        print(tfidf_matrix.shape)

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        print(cosine_sim.shape)

        return cosine_sim

    # Creates indices given all the titles of the dataset, adn then returns those indices 
    def getIndices(self, moviesDF):
        index = moviesDF["title_y"]
        indices = pd.Series(moviesDF.index, index).drop_duplicates()
        print(indices.head())
        return indices

    # Takes the cosine score of a specific movie and sort them by cosine score. 
    # Select the following 10 values because the first entry is itself. 
    # Retrieves those movie indices, maps it to titles, and return the movie list.
    def getRecommendations(self, title, cosine_sim, indices):

        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        movies_indices = [ind[0] for ind in sim_scores]
        movies = self.moviesDF["title_y"].iloc[movies_indices]
        print(movies)
        return movies

    #Running the program
    def runProgram(self, cos_sim, indices):
        print("##########    CONTENT-BASED FILTERING    ###########")
        title = input("Enter the name of the movie that you would like to see the Recomendations: ")
        print("Recommendations for " + title)
        self.getRecommendations(title, cos_sim, indices)

# Main
if __name__ == "__main__":
    creditsFileName = sys.argv[1]
    moviesFileName = sys.argv[2]
    creditsDF = ""
    moviesDF = ""
    myMovieRecommender = MovieRecommender(creditsFileName, moviesFileName, creditsDF, moviesDF)
    myMovieRecommender.readDatasets(creditsFileName, moviesFileName)
    myMovieRecommender.calculateAverageAndQuantiles(myMovieRecommender.moviesDF)
    myMovieRecommender.plot(myMovieRecommender.moviesDF)
    cos_sim = myMovieRecommender.cosineSimilarity(myMovieRecommender.moviesDF)
    indices = myMovieRecommender.getIndices(myMovieRecommender.moviesDF)
    myMovieRecommender.runProgram(cos_sim, indices)






