import os
import string
import json
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from config import ROOT_DIR


# Ensure necessary NLTK data files are downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# File paths
INPUT_FILE           = os.path.join(ROOT_DIR, "data", "synthetic_biology_corpus.json")  # Change to .txt if you saved as text
PREPROCESSED_FILE    = os.path.join(ROOT_DIR, "data", "synthetic_biology_preprocessed.json")
VECTORIZER_FILE      = os.path.join(ROOT_DIR, "data", "tfidf_vectorizer.pkl")
VECTORIZED_DATA_FILE = os.path.join(ROOT_DIR, "data", "tfidf_vectors.pkl")

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_corpus(filepath):
    """Load sentences from a JSON or text file."""
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                sentences = json.load(file)
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                sentences = [line.strip() for line in file if line.strip()]
        else:
            print("Unsupported file format. Please use .json or .txt")
            return []
        print(f"Loaded {len(sentences)} sentences.")
        return sentences
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return []

def preprocess_sentence(sentence):
    """Clean and preprocess a single sentence."""
    # Lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(sentence)
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Rejoin into a single string
    return ' '.join(tokens)

def preprocess_corpus(sentences):
    """Preprocess a list of sentences."""
    preprocessed = []
    for sentence in sentences:
        processed = preprocess_sentence(sentence)
        if processed:  # Ensure sentence is not empty after preprocessing
            preprocessed.append(processed)
    print(f"Preprocessed {len(preprocessed)} sentences.")
    return preprocessed

def save_preprocessed_corpus(preprocessed_sentences, filepath):
    """Save preprocessed sentences to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(preprocessed_sentences, file, ensure_ascii=False, indent=2)
        print(f"Saved preprocessed corpus to {filepath}.")
    except Exception as e:
        print(f"Error saving preprocessed corpus: {e}")

def vectorize_corpus(preprocessed_sentences, vectorizer_path, data_path):
    """Convert preprocessed text to TF-IDF vectors and save the vectorizer and vectors."""
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(preprocessed_sentences)
    # Save the vectorizer
    with open(vectorizer_path, 'wb') as file:
        pickle.dump(vectorizer, file)
    print(f"Saved TF-IDF vectorizer to {vectorizer_path}.")
    # Save the vectors
    with open(data_path, 'wb') as file:
        pickle.dump(tfidf_vectors, file)
    print(f"Saved TF-IDF vectors to {data_path}.")

def main():
    # Load corpus
    sentences = load_corpus(INPUT_FILE)
    if not sentences:
        print("No sentences to process.")
        return
    # Preprocess sentences
    preprocessed = preprocess_corpus(sentences)
    if not preprocessed:
        print("No sentences left after preprocessing.")
        return
    # Save preprocessed corpus
    save_preprocessed_corpus(preprocessed, PREPROCESSED_FILE)
    # Vectorize and save TF-IDF vectors
    vectorize_corpus(preprocessed, VECTORIZER_FILE, VECTORIZED_DATA_FILE)

if __name__ == "__main__":
    main()
