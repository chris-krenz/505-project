# classify_sentence.py
import os
import json
import pickle
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_model(model_path):
    """Load the trained Logistic Regression model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        exit(1)


def load_vectorizer(vectorizer_path):
    """Load the fitted TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"Loaded vectorizer from {vectorizer_path}.")
        return vectorizer
    except Exception as e:
        print(f"Error loading vectorizer from {vectorizer_path}: {e}")
        exit(1)


def load_keywords(keywords_path):
    """Load the list of keywords to remove from sentences."""
    try:
        with open(keywords_path, 'r', encoding='utf-8') as file:
            keywords_dict = json.load(file)
        keywords = list(keywords_dict.values())
        print(f"Loaded {len(keywords)} keywords from {keywords_path}.")
        return keywords
    except Exception as e:
        print(f"Error loading keywords from {keywords_path}: {e}")
        exit(1)


def remove_keywords(sentence, keywords):
    """Remove all keywords from the sentence (case-insensitive)."""
    for keyword in keywords:
        sentence = sentence.replace(keyword, "")
        sentence = sentence.replace(keyword.lower(), "")
        sentence = sentence.replace(keyword.upper(), "")
    return sentence


def classify_sentence(model, vectorizer, keywords, sentence):
    """Preprocess and classify the input sentence."""
    processed_sentence = remove_keywords(sentence, keywords)
    vectorized = vectorizer.transform([processed_sentence])
    prediction = model.predict(vectorized)[0]
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Manual Sentence Classifier")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained Logistic Regression model (.pkl file).'
    )
    parser.add_argument(
        '--vectorizer',
        type=str,
        required=True,
        help='Path to the fitted TF-IDF vectorizer (.pkl file).'
    )
    parser.add_argument(
        '--keywords',
        type=str,
        required=True,
        help='Path to the JSON file mapping labels to keywords.'
    )
    args = parser.parse_args()

    # Load resources
    model = load_model(args.model)
    vectorizer = load_vectorizer(args.vectorizer)
    keywords = load_keywords(args.keywords)

    print("\n=== Manual Sentence Classifier ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a sentence to classify: ")
        if user_input.lower() == 'exit':
            print("Exiting classifier.")
            break
        if not user_input.strip():
            print("Empty input. Please enter a valid sentence.\n")
            continue

        predicted_label = classify_sentence(model, vectorizer, keywords, user_input)
        print(f"Predicted Label: {predicted_label}\n")


if __name__ == "__main__":
    main()
    