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
        #self.readDatasets(filename1, filename2)

    # Method to read the datasets
    def readDatasets(self, filename1, filename2):
        self.creditsDF = pd.read_csv(filename1, index_col=0)
        self.moviesDF = pd.read_csv(filename2, index_col=0)
        print(self.moviesDF.head())
        print(self.creditsDF.head())
        self.moviesDF = pd.merge(self.creditsDF, self.moviesDF, on = "id", how = "inner")
        #return moviesDF


    def calculateAverageAndQuantiles(self, moviesDF):
        averageVote = moviesDF["vote_average"].mean()
        quantile90percent = moviesDF["vote_count"].quantile(0.9)
        print("Average Vote: ", averageVote)
        print("Quantile 90%: ", quantile90percent)

        moviesDF = moviesDF.copy().loc[moviesDF["vote_count"] >= quantile90percent]
        print(moviesDF.shape)
        #print(self.weighted_rating(newMoviesDF, averageVote, quantile90percent))
        #self.weighted_rating(newMoviesDF, averageVote, quantile90percent)
        self.updateAndPrint(moviesDF, averageVote, quantile90percent)
        
    

    def weighted_rating(self, x, averageVote, quantile90percent):
        voteCount = x["vote_count"]
        R = x["vote_average"]
        #print("R = " + R)
        return (voteCount/(voteCount + quantile90percent) * R) + (quantile90percent/(voteCount + quantile90percent) * averageVote)

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


    def cosineSimilarity(self, moviesDF):
        print(moviesDF["overview"].head(5))

        tfidf = TfidfVectorizer(stop_words="english")
        moviesDF["overview"] = moviesDF["overview"].fillna("")

        tfidf_matrix = tfidf.fit_transform(moviesDF["overview"])
        print(tfidf_matrix.shape)

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        print(cosine_sim.shape)

        return cosine_sim

    def getIndices(self, moviesDF):
        index = moviesDF["title_y"]
        indices = pd.Series(moviesDF.index, index).drop_duplicates()
        print(indices.head())
        return indices
        #indices = pd.Series(moviesDF.index, index=moviesDF["title_y"]).drop_duplicates()
        #print(indices.head())
        #return indices

    def getRecommendations(self, title, cosine_sim, indices):
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
        movies = self.moviesDF["title_y"].iloc[movies_indices]
        print(movies)
        return movies

    #Runnig the program
    def runProgram(self, cos_sim, indices):
        print("##########    CONTENT-BASED FILTERING    ###########")
        title = input("Enter the name of the movie that you would like to see the Recomendations: ")
        print("Recommendations for " + title)
        self.getRecommendations(title, cos_sim, indices)

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






