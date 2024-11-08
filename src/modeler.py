import os
import json
import pickle

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config import ROOT_DIR

# File paths
PREPROCESSED_FILE     = os.path.join(ROOT_DIR, "data", "synthetic_biology_preprocessed.json")
TOPIC_MODEL_FILE      = os.path.join(ROOT_DIR, "data", "lda_model.pkl")
COUNT_VECTORIZER_FILE = os.path.join(ROOT_DIR, "data", "count_vectorizer.pkl")
N_TOPICS    = 10  # Adjust based on your needs
TOP_N_WORDS = 10  # Number of top words to display per topic


def load_corpus(filepath):
    """Load preprocessed sentences from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            sentences = json.load(file)
        print(f"Loaded {len(sentences)} preprocessed sentences.")
        return sentences
    except Exception as e:
        print(f"Error loading preprocessed corpus: {e}")
        return []

def vectorize_corpus(sentences):
    """Vectorize the corpus using CountVectorizer."""
    vectorizer = CountVectorizer()
    term_matrix = vectorizer.fit_transform(sentences)
    print("Vectorized the corpus using CountVectorizer.")
    return vectorizer, term_matrix

def train_lda(term_matrix, n_topics):
    """Train an LDA model."""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(term_matrix)
    print(f"Trained LDA model with {n_topics} topics.")
    return lda

def save_model(model, filepath):
    """Save the model to a pickle file."""
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Saved model to {filepath}.")
    except Exception as e:
        print(f"Error saving model: {e}")

def display_topics(model, feature_names, no_top_words):
    """Display the top words for each topic."""
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx +1}:")
        top_indices = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        print(", ".join(top_words))
        print()

def main():
    # Load corpus
    sentences = load_corpus(PREPROCESSED_FILE)
    if not sentences:
        print("No sentences to process.")
        return
    # Vectorize corpus
    vectorizer, term_matrix = vectorize_corpus(sentences)
    # Train LDA model
    lda_model = train_lda(term_matrix, N_TOPICS)
    # Save models
    save_model(lda_model, TOPIC_MODEL_FILE)
    save_model(vectorizer, COUNT_VECTORIZER_FILE)
    # Display topics
    feature_names = vectorizer.get_feature_names_out()
    display_topics(lda_model, feature_names, TOP_N_WORDS)

if __name__ == "__main__":
    main()
