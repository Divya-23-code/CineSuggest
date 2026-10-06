import os 
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# CINE SUGGEST - MOVIE RECOMMENDATION SYSTEM
# ---------------------------------------------------

st.set_page_config(
    page_title="CineSuggest",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "IMDbRatings_IndianMovies.csv")
    df = pd.read_csv(csv_path)
    # Remove unnecessary spaces from column names
    df.columns = df.columns.str.strip()

    # Fill missing values
    text_columns = [
        "Name", "Genre", "Director",
        "Actor 1", "Actor 2", "Actor 3"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("")

    # Convert Rating and Year to numbers
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Clean movie names
    df["Name"] = df["Name"].str.strip()

    return df


df = load_data()


# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

# Create a combined feature column
df["combined_features"] = (
    df["Genre"].astype(str) + " " +
    df["Director"].astype(str) + " " +
    df["Actor 1"].astype(str) + " " +
    df["Actor 2"].astype(str) + " " +
    df["Actor 3"].astype(str)
)

# TF-IDF converts text into numerical features
tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(df["combined_features"])


# Calculate similarity between movies


# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------

def recommend_movies(movie_name, number_of_movies=10):

    # Find movie index
    matching_movies = df[
        df["Name"].str.lower() == movie_name.lower()
    ]

    if matching_movies.empty:
        return None

    movie_index = matching_movies.index[0]

    # Get similarity scores
    similarity_scores = list(
    enumerate(
        cosine_similarity(
            tfidf_matrix[movie_index],
            tfidf_matrix
        )[0]
    )
)

    # Sort movies according to similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get top recommendations
    recommended_indices = [
        index
        for index, score in similarity_scores[1:number_of_movies + 1]
    ]

    recommendations = df.iloc[recommended_indices].copy()

    return recommendations


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🎬 CineSuggest")

st.subheader("Your Personal Indian Movie Recommendation System")

st.write(
    "Discover movies similar to the ones you love "
    "using genre, director and actor information."
)

st.divider()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🎬 CineSuggest")

st.sidebar.write(
    "A content-based movie recommendation system "
    "built using Python and Machine Learning."
)

st.sidebar.divider()

st.sidebar.write("### Dataset Information")

st.sidebar.write(
    f"🎞️ Movies: {len(df)}"
)

st.sidebar.write(
    f"⭐ Rated Movies: {df['Rating'].notna().sum()}"
)

st.sidebar.write(
    f"🎬 Directors: {df['Director'].nunique()}"
)


# ---------------------------------------------------
# MOVIE SELECTION
# ---------------------------------------------------

st.header("🔎 Find Similar Movies")

movie_list = sorted(
    df["Name"].dropna().unique().tolist()
)

selected_movie = st.selectbox(
    "Select a movie you like:",
    movie_list
)

number_of_movies = st.slider(
    "Number of recommendations:",
    min_value=5,
    max_value=15,
    value=10
)


# ---------------------------------------------------
# RECOMMEND BUTTON
# ---------------------------------------------------

if st.button("✨ Recommend Movies"):

    recommendations = recommend_movies(
        selected_movie,
        number_of_movies
    )

    if recommendations is None:

        st.error("Movie not found in the dataset.")

    else:

        st.success(
            f"Movies similar to **{selected_movie}**:"
        )

        # Display recommendations
        for _, movie in recommendations.iterrows():

            movie_name = movie["Name"]

            year = movie["Year"]

            genre = movie["Genre"]

            rating = movie["Rating"]

            director = movie["Director"]

            actors = ", ".join(
                filter(
                    None,
                    [
                        movie["Actor 1"],
                        movie["Actor 2"],
                        movie["Actor 3"]
                    ]
                )
            )

            st.markdown(
                f"### 🎬 {movie_name}"
            )

            col1, col2 = st.columns(2)

            with col1:

                if pd.notna(year):
                    st.write(f"📅 **Year:** {int(year)}")
                else:
                    st.write("📅 **Year:** Not available")

                st.write(
                    f"🎭 **Genre:** {genre if genre else 'Not available'}"
                )

            with col2:

                if pd.notna(rating):
                    st.write(
                        f"⭐ **IMDb Rating:** {rating}/10"
                    )
                else:
                    st.write(
                        "⭐ **IMDb Rating:** Not available"
                    )

                st.write(
                    f"🎥 **Director:** "
                    f"{director if director else 'Not available'}"
                )

            if actors:
                st.write(f"👥 **Actors:** {actors}")

            st.divider()


# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

with st.expander("📊 View Dataset"):

    st.dataframe(
        df[
            [
                "Name",
                "Year",
                "Genre",
                "Rating",
                "Director",
                "Actor 1",
                "Actor 2",
                "Actor 3"
            ]
        ],
        use_container_width=True
    )


# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

st.divider()

st.header("ℹ️ About CineSuggest")

st.write(
    """
    CineSuggest is a content-based movie recommendation system.
    
    The system uses movie information such as genre, director,
    and actors to find movies that are similar to the movie
    selected by the user.
    
    TF-IDF is used to convert textual movie information into
    numerical vectors, and cosine similarity is used to
    calculate the similarity between movies.
    """
)
