"""
TASK 3: Recommendation System
CODSOFT AI Internship

A content-based movie recommendation system. Recommends movies similar
to a given movie based on genre and description similarity, using
TF-IDF vectorization and Cosine Similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self):
        self.data = self._create_dataset()
        self.data['combined_features'] = (
            self.data['genre'] + ' ' + self.data['description']
        )
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_features'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _create_dataset(self):
        # Small built-in dataset so this runs with zero external files
        movies = [
            {"title": "The Matrix", "genre": "Sci-Fi Action",
             "description": "A hacker discovers reality is a simulation and joins a rebellion against machines."},
            {"title": "Inception", "genre": "Sci-Fi Thriller",
             "description": "A thief who steals secrets through dream-sharing technology takes on a mind-bending heist."},
            {"title": "Interstellar", "genre": "Sci-Fi Drama",
             "description": "A team of explorers travel through a wormhole in space to save humanity."},
            {"title": "The Dark Knight", "genre": "Action Crime",
             "description": "Batman faces the Joker, a criminal mastermind who wants to plunge Gotham into chaos."},
            {"title": "John Wick", "genre": "Action Thriller",
             "description": "A retired hitman seeks vengeance after the murder of his dog and theft of his car."},
            {"title": "The Notebook", "genre": "Romance Drama",
             "description": "A couple falls deeply in love but is separated by class differences and war."},
            {"title": "La La Land", "genre": "Romance Musical",
             "description": "A jazz musician and an actress pursue their dreams while falling in love in Los Angeles."},
            {"title": "Titanic", "genre": "Romance Drama",
             "description": "A love story unfolds aboard the ill-fated ship on its maiden voyage."},
            {"title": "Get Out", "genre": "Horror Thriller",
             "description": "A young man uncovers a disturbing secret when he visits his girlfriend's family estate."},
            {"title": "A Quiet Place", "genre": "Horror Sci-Fi",
             "description": "A family must live in silence to avoid alerting creatures that hunt by sound."},
            {"title": "The Conjuring", "genre": "Horror Mystery",
             "description": "Paranormal investigators help a family terrorized by a dark presence in their farmhouse."},
            {"title": "Toy Story", "genre": "Animation Family",
             "description": "Toys come to life and go on an adventure to be reunited with their owner."},
            {"title": "Finding Nemo", "genre": "Animation Family",
             "description": "A clownfish embarks on a journey across the ocean to find his lost son."},
            {"title": "The Avengers", "genre": "Action Sci-Fi",
             "description": "Superheroes team up to save Earth from an alien invasion led by Loki."},
            {"title": "Guardians of the Galaxy", "genre": "Action Sci-Fi Comedy",
             "description": "A group of misfit space outlaws band together to save the galaxy."},
        ]
        return pd.DataFrame(movies)

    def recommend(self, movie_title, top_n=5):
        matches = self.data[self.data['title'].str.lower() == movie_title.lower()]
        if matches.empty:
            print(f"Movie '{movie_title}' not found in database.")
            print("Available movies:", ", ".join(self.data['title'].tolist()))
            return []

        idx = matches.index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Skip the first result (it's the movie itself)
        top_matches = similarity_scores[1:top_n + 1]

        recommendations = []
        for i, score in top_matches:
            recommendations.append({
                "title": self.data.iloc[i]['title'],
                "genre": self.data.iloc[i]['genre'],
                "similarity": round(score, 3)
            })
        return recommendations

    def list_movies(self):
        print("\nAvailable movies in database:")
        for title in self.data['title'].tolist():
            print(f"  - {title}")
        print()


def main():
    recommender = MovieRecommender()
    print("=== Movie Recommendation System ===")
    recommender.list_movies()

    while True:
        movie = input("Enter a movie name to get recommendations (or 'quit' to exit): ")
        if movie.lower() == 'quit':
            print("Goodbye!")
            break

        recommendations = recommender.recommend(movie)
        if recommendations:
            print(f"\nBecause you liked '{movie}', you might also enjoy:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['title']} ({rec['genre']}) - similarity: {rec['similarity']}")
            print()


if __name__ == "__main__":
    main()
