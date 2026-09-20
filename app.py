import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Page settings
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie!")

# Load movie data
movies = pd.read_csv("tmdb_5000_movies.csv")

# Select useful columns
movies = movies[["title", "overview", "genres", "keywords"]]

# Handle missing values
movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].fillna("")
movies["keywords"] = movies["keywords"].fillna("")

# Combine movie information
movies["combined_features"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"]
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")

feature_matrix = vectorizer.fit_transform(
    movies["combined_features"]
)

# Find similar movies efficiently
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(feature_matrix)


def recommend_movies(movie_title):

    movie_index = movies[
        movies["title"] == movie_title
    ].index[0]

    movie_vector = feature_matrix[movie_index]

    distances, indices = model.kneighbors(
        movie_vector,
        n_neighbors=6
    )

    recommendations = []

    for index in indices[0][1:]:
        recommendations.append(
            movies.iloc[index]["title"]
        )

    return recommendations


# Movie selection
selected_movie = st.selectbox(
    "🎥 Select a movie:",
    movies["title"].values
)

# Recommendation button
if st.button("🎯 Recommend Movies"):

    recommendations = recommend_movies(selected_movie)

    st.subheader("🍿 You may also like:")

    for i, movie in enumerate(recommendations, 1):
        st.write(f"**{i}. {movie}**")