# classify_sentence.py
import os
import json
import pickle
import argparse
import sys


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


def interactive_mode(model, vectorizer, keywords):
    """Interactive mode for single or multiple sentence classification."""
    print("\n=== Manual Sentence Classifier ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter sentence(s) to classify (separated by ';') or type 'exit' to quit:\n")
        if user_input.lower() == 'exit':
            print("Exiting classifier.")
            break
        if not user_input.strip():
            print("Empty input. Please enter valid sentence(s).\n")
            continue
        
        # Split input into individual sentences based on semicolon delimiter
        sentences = [s.strip() for s in user_input.split(';') if s.strip()]
        if not sentences:
            print("No valid sentences found. Please try again.\n")
            continue
        
        # Classify each sentence
        for idx, sentence in enumerate(sentences, 1):
            predicted_label = classify_sentence(model, vectorizer, keywords, sentence)
            print(f"Sentence {idx}: '{sentence}'")
            print(f"Predicted Label: {predicted_label}\n")


def batch_mode(model, vectorizer, keywords, input_file):
    """Batch mode for classifying multiple sentences from a file."""
    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' does not exist.")
        exit(1)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        if not sentences:
            print(f"No valid sentences found in '{input_file}'.")
            exit(1)
        
        print(f"\n=== Batch Sentence Classifier ===")
        print(f"Classifying {len(sentences)} sentences from '{input_file}'.\n")
        
        for idx, sentence in enumerate(sentences, 1):
            predicted_label = classify_sentence(model, vectorizer, keywords, sentence)
            print(f"Sentence {idx}: '{sentence}'")
            print(f"Predicted Label: {predicted_label}\n")
    
    except Exception as e:
        print(f"An error occurred while reading '{input_file}': {e}")
        exit(1)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Manual and Batch Sentence Classifier")
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
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Path to a text file containing sentences to classify (one per line). If not provided, the script runs in interactive mode.'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Load resources
    model = load_model(args.model)
    vectorizer = load_vectorizer(args.vectorizer)
    keywords = load_keywords(args.keywords)
    
    if args.input_file:
        # Batch mode
        batch_mode(model, vectorizer, keywords, args.input_file)
    else:
        # Interactive mode
        interactive_mode(model, vectorizer, keywords)


if __name__ == "__main__":
    main()
    