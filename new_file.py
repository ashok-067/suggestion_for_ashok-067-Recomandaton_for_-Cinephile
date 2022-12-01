import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from csv file into pandas data frame

df=pd.read_csv("movies.csv")

#selecting the revelant features
selected_features=["genres","keywords","tagline","cast","director"]

#filling the null values in the selected features
for values in selected_features:
    df[values]=df[values].fillna("")
    
#combine all the selected features
combine_data=df["genres"]+" "+df["keywords"]+" "+df["tagline"]+" "+df["cast"]+" "+df["director"]

#converting the text to the feature vectors
featurizer=TfidfVectorizer()
feture_vector=featurizer.fit_transform(combine_data)

# print(feture_vector)
"""COSINE SIMILARITIES"""

#getting the similarites scores using the cosine similarites
similarities=cosine_similarity(feture_vector)

#getting the movie name from the user
movie_name=input("pls enter movie name:")

#craeting the list with all movie names given in the data set
all_movies_list=df["original_title"].tolist()

#finding the close match for the movie name given by the user
find_close_match=difflib.get_close_matches(movie_name,all_movies_list)
close_match=find_close_match[0]

#finding the index of the movie with original_title
index_ofthe_movie=df[df.original_title== close_match]["index"].values[0]
index_ofthe_movie

#getting the names of the similar movies
similarity_score=list(enumerate(similarities[index_ofthe_movie]))

#sorting the movies based on their similarites score
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

#print the names of the moies based on thier similar scores
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=df[df.index==index]["original_title"].values[0]
    if i<11 :
        print(i, ".",title_from_index)
        i+=1
    