# benchmarker.py
import os
import json
import argparse
import pickle
import math
import re
import numpy as np
from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split


from config import ROOT_DIR


def load_data(preprocessed_file, labels_file):
    try:
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences from {preprocessed_file}.")
    except Exception as e:
        print(f"Error loading {preprocessed_file}: {e}")
        return [], []

    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} labels from {labels_file}.")
    except Exception as e:
        print(f"Error loading {labels_file}: {e}")
        return [], []

    if len(sentences) != len(labels):
        print("Mismatch between number of sentences and labels.")
        return [], []

    single_labels = [label_list[0] if label_list else "Other" for label_list in labels]

    return sentences, single_labels


def load_label_to_keywords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            keyword_label_map = json.load(file)
        label_to_keywords = {}
        for keyword, label in keyword_label_map.items():
            label_to_keywords.setdefault(label, []).append(keyword)
        print(f"Created label-to-keywords mapping with {len(label_to_keywords)} labels.")
        return label_to_keywords
    except Exception as e:
        print(f"Error loading label-to-keywords mapping from {filepath}: {e}")
        exit(1)


def analyze_label_distribution(labels):
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)

    problematic_classes = [label for label, count in label_counts.items() if count < 2]
    print(f"Classes with fewer than 2 instances: {problematic_classes}")

    return problematic_classes


def remove_rare_classes(sentences, labels, problematic_classes):
    filtered_sentences = []
    filtered_labels = []
    for sentence, label in zip(sentences, labels):
        if label not in problematic_classes:
            filtered_sentences.append(sentence)
            filtered_labels.append(label)
    print(f"Removed {len(sentences) - len(filtered_sentences)} instances from problematic classes.")
    return filtered_sentences, filtered_labels


def remove_keywords(sentences, labels, label_to_keywords):
    modified_sentences = []
    removal_count = 0
    for sentence, label in zip(sentences, labels):
        keywords = label_to_keywords.get(label, [])
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            sentence, num_subs = pattern.subn("", sentence)
            removal_count += num_subs
        modified_sentences.append(sentence)
    print(f"Removed {removal_count} keywords from sentences based on labels.")
    return modified_sentences


def build_vocabulary(sentences, max_features=1000):
    term_freq = Counter()
    for sentence in sentences:
        terms = sentence.split()
        term_freq.update(terms)
    most_common = term_freq.most_common(max_features)
    vocab = {term: idx for idx, (term, _) in enumerate(most_common)}
    print(f"Built vocabulary with {len(vocab)} terms.")
    return vocab


def compute_idf(sentences, vocab):
    N = len(sentences)
    df = np.zeros(len(vocab))
    for sentence in sentences:
        terms = set(sentence.split())
        for term in terms:
            if term in vocab:
                df[vocab[term]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1
    print("Computed IDF for vocabulary.")
    return idf


def vectorize_sentences(sentences, vocab, idf):
    tfidf_matrix = np.zeros((len(sentences), len(vocab)))
    for i, sentence in enumerate(sentences):
        terms = sentence.split()
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            if term in vocab:
                idx = vocab[term]
                tf = count / len(terms)  # Term Frequency
                tfidf_matrix[i][idx] = tf * idf[idx]
        if (i + 1) % 100 == 0 or (i + 1) == len(sentences):
            print(f"Vectorized sentence {i + 1}/{len(sentences)}.")
    return tfidf_matrix


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    normalized_vectors = vectors / norms
    print("Normalized TF-IDF vectors.")
    return normalized_vectors


def train_logistic_regression(X_train, y_train, unique_labels, learning_rate=0.1, epochs=1000):
    num_features = X_train.shape[1]
    classifiers = {}
    X_train = X_train.T  # ...shape: (features, samples)
    y_train_array = np.array(y_train)

    for label in unique_labels:
        print(f"Training classifier for label: {label}")
        binary_y = np.where(y_train_array == label, 1, 0)  # Shape:  (samples, )

        weights = np.zeros(num_features)
        bias = 0.0

        for epoch in range(1, epochs + 1):
            linear_output = np.dot(weights, X_train) + bias  # Shape: (samples,)
            predictions = 1 / (1 + np.exp(-linear_output))  # Shape: (samples,)

            error = binary_y - predictions  # Shape: (samples,)

            gradient_weights = np.dot(X_train, error)  # Shape: (features,)
            gradient_bias = np.sum(error)  # Scalar

            weights += learning_rate * gradient_weights
            bias += learning_rate * gradient_bias

            if epoch % 100 == 0 or epoch == epochs:
                loss = -np.mean(binary_y * np.log(predictions + 1e-15) + (1 - binary_y) * np.log(1 - predictions + 1e-15))
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

        classifiers[label] = {
            "weights": weights.tolist(),
            "bias": bias
        }
        print(f"Finished training classifier for label: {label}\n")

    return classifiers


def predict(classifiers, X):
    predictions = []
    X = X.T 
    for i in range(X.shape[1]):
        sample = X[:, i]  
        scores = {}
        for label, params in classifiers.items():
            weights = np.array(params["weights"])
            bias = params["bias"]
            linear_output = np.dot(weights, sample) + bias
            prediction = 1 / (1 + np.exp(-linear_output))
            scores[label] = prediction
        predicted_label = max(scores, key=scores.get)
        predictions.append(predicted_label)
        if (i + 1) % 100 == 0 or (i + 1) == X.shape[1]:
            print(f"Predicted label for sample {i + 1}/{X.shape[1]}.")
    return predictions


def evaluate_model(y_true, y_pred, unique_labels):

    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)
    print(f"\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}\n")


    metrics = {
        "accuracy": accuracy,
        "precision": {},
        "recall": {},
        "f1_score": {},
        "support": {},
        "confusion_matrix": {}
    }

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        i = label_to_index[yt]
        j = label_to_index[yp]
        confusion_matrix[i][j] += 1

    table_data = []
    for label in unique_labels:
        idx = label_to_index[label]
        tp = confusion_matrix[idx][idx]
        fp = np.sum(confusion_matrix[:, idx]) - tp
        fn = np.sum(confusion_matrix[idx, :]) - tp
        support = np.sum(confusion_matrix[idx, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["precision"][label] = precision
        metrics["recall"][label] = recall
        metrics["f1_score"][label] = f1
        metrics["support"][label] = support

        table_data.append([label, precision, recall, f1, support])

    label_width = max(len(label) for label in unique_labels) + 2
    precision_width = 10
    recall_width = 10
    f1_width = 10
    support_width = 8

    header = (
        f"{'Label':<{label_width}}"
        f"{'Precision':<{precision_width}}"
        f"{'Recall':<{recall_width}}"
        f"{'F1-Score':<{f1_width}}"
        f"{'Support':<{support_width}}"
    )
    separator = "-" * (label_width + precision_width + recall_width + f1_width + support_width)
    print(header)
    print(separator)

    for row in table_data:
        label, precision, recall, f1, support = row
        print(
            f"{label:<{label_width}}"
            f"{precision:<{precision_width}.4f}"
            f"{recall:<{recall_width}.4f}"
            f"{f1:<{f1_width}.4f}"
            f"{support:<{support_width}}"
        )

    metrics["confusion_matrix"] = confusion_matrix.tolist()
    print("\nConfusion Matrix:")
    print("Labels:", unique_labels)
    for row in confusion_matrix:
        print(row)

    return metrics


def save_metrics(metrics, filepath="model_metrics_manual.json"):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved evaluation metrics to {filepath}.")
    except Exception as e:
        print(f"Error saving metrics to {filepath}: {e}")


def save_classifiers(classifiers, filepath="manual_logistic_regression_model.pkl"):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(classifiers, f)
        print(f"Saved classifiers to {filepath}.")
    except Exception as e:
        print(f"Error saving classifiers to {filepath}: {e}")


def save_vocabulary(vocab, idf, filepath="vocab_idf.pkl"):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({'vocab': vocab, 'idf': idf}, f)
        print(f"Saved vocabulary and IDF to {filepath}.")
    except Exception as e:
        print(f"Error saving vocabulary and IDF to {filepath}: {e}")


def parse_arguments():

    parser = argparse.ArgumentParser(description="Manual Benchmarking of TF-IDF and Logistic Regression")
    parser.add_argument(
        '--remove-keywords',
        action='store_true',
        help='Flag to remove keywords from sentences based on their labels.'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    PREPROCESSED_FILE = os.path.join(ROOT_DIR, "data", "synthetic_biology_preprocessed.json")
    LABELS_FILE = os.path.join(ROOT_DIR, "data", "labels.json")
    KEYWORDS_PATH = os.path.join(ROOT_DIR, "src", "KEYWORD_LABEL_MAP.json")

    label_to_keywords = load_label_to_keywords(KEYWORDS_PATH)

    sentences, labels = load_data(PREPROCESSED_FILE, LABELS_FILE)
    if not sentences or not labels:
        print("Data loading failed. Exiting.")
        return

    problematic_classes = analyze_label_distribution(labels)

    if problematic_classes:
        sentences, labels = remove_rare_classes(sentences, labels, problematic_classes)

    analyze_label_distribution(labels)

    sentences_filtered = []
    labels_filtered = []
    for sentence, label in zip(sentences, labels):
        if label != "Other":
            sentences_filtered.append(sentence)
            labels_filtered.append(label)
    print(f"Excluded 'Other' category. Remaining instances: {len(sentences_filtered)}")

    if args.remove_keywords:
        sentences_filtered = remove_keywords(sentences_filtered, labels_filtered, label_to_keywords)

    if not sentences_filtered:
        print("No sentences left after filtering. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        sentences_filtered,
        labels_filtered,
        test_size=0.2,
        random_state=42,
        stratify=labels_filtered
    )
    print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples.")

    vocab = build_vocabulary(X_train, max_features=1000)

    idf = compute_idf(X_train, vocab)

    X_train_tfidf = vectorize_sentences(X_train, vocab, idf)
    X_test_tfidf = vectorize_sentences(X_test, vocab, idf)
    print(f"Vectorized training and testing data.")

    X_train_tfidf = normalize_vectors(X_train_tfidf)
    X_test_tfidf = normalize_vectors(X_test_tfidf)
    print(f"Normalized TF-IDF vectors.")

    unique_labels = list(set(y_train))
    print(f"Unique labels: {unique_labels}")

    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    classifiers = train_logistic_regression(X_train_tfidf, y_train_np, unique_labels, learning_rate=0.1, epochs=1000)

    y_pred = predict(classifiers, X_test_tfidf)

    metrics = evaluate_model(y_test_np, y_pred, unique_labels)

    save_metrics(metrics, filepath="model_metrics_manual.json")

    save_classifiers(classifiers, filepath="manual_logistic_regression_model.pkl")

    save_vocabulary(vocab, idf, filepath="vocab_idf.pkl")


if __name__ == "__main__":
    main()
