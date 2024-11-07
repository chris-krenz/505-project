import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

def load_json(filepath):
    """Load JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {filepath}.")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def save_model(model, vectorizer, model_path, vectorizer_path):
    """Save the trained model and vectorizer to disk."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {model_path}.")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
    
    try:
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Saved vectorizer to {vectorizer_path}.")
    except Exception as e:
        print(f"Error saving vectorizer to {vectorizer_path}: {e}")

def vectorize_text(X_train, X_test, max_features=1000):
    """Convert text data into TF-IDF feature vectors."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vectorized text with TF-IDF. Number of features: {max_features}")
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    print("Trained Logistic Regression classifier.")
    return clf

def train_dummy_classifier(X_train, y_train, strategy='most_frequent'):
    """Train a Dummy classifier as a baseline."""
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    dummy_clf.fit(X_train, y_train)
    print(f"Trained Dummy classifier with strategy='{strategy}'.")
    return dummy_clf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a trained model on the test set and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }

def filter_out_other(sentences, labels, other_label="Other"):
    """
    Filter out instances labeled as 'Other'.
    
    Args:
        sentences (list): List of sentences.
        labels (list): List of labels corresponding to each sentence.
        other_label (str): The label to exclude.
        
    Returns:
        filtered_sentences (list): Sentences without the 'Other' label.
        filtered_labels (list): Corresponding labels without the 'Other' label.
    """
    filtered_sentences = []
    filtered_labels = []
    for sentence, label in zip(sentences, labels):
        if label != other_label:
            filtered_sentences.append(sentence)
            filtered_labels.append(label)
    print(f"Filtered out {len(sentences) - len(filtered_sentences)} 'Other' instances.")
    return filtered_sentences, filtered_labels

def main():
    # Define file paths
    PREPROCESSED_FILE = "synthetic_biology_preprocessed.json"
    GROUND_TRUTH_LABELS_FILE = "labels.json"
    LOG_REG_MODEL_PATH = "logistic_regression_model.pkl"
    DUMMY_MODEL_PATH = "dummy_classifier.pkl"
    VECTORIZER_PATH = "tfidf_vectorizer.pkl"

    # Load data
    sentences = load_json(PREPROCESSED_FILE)
    ground_truth_labels = load_json(GROUND_TRUTH_LABELS_FILE)

    if not sentences or not ground_truth_labels:
        print("Failed to load sentences or labels. Exiting.")
        return

    if len(sentences) != len(ground_truth_labels):
        print("Mismatch between number of sentences and labels. Exiting.")
        return

    # Flatten labels: take the first label from each list
    y_true = [label_list[0] if label_list else "Other" for label_list in ground_truth_labels]

    # Analyze label distribution
    label_counts = Counter(y_true)
    print("Label distribution:", label_counts)

    # Identify problematic classes with fewer than 2 instances
    problematic_classes = [label for label, count in label_counts.items() if count < 2]
    print(f"Classes with fewer than 2 instances: {problematic_classes}")

    # Option A: Merge problematic classes into 'Other'
    if problematic_classes:
        y_true = [label if label not in problematic_classes else "Other" for label in y_true]
        print("Merged rare classes into 'Other'.")
    
    # Re-analyze label distribution after merging
    label_counts = Counter(y_true)
    print("Updated label distribution:", label_counts)

    # Split data into training and testing sets with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
        print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples.")
    except ValueError as ve:
        print(f"Error during train_test_split: {ve}")
        return

    # Vectorize text
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test, max_features=1000)

    # Train Logistic Regression
    log_reg_clf = train_logistic_regression(X_train_tfidf, y_train)

    # Train Dummy Classifier
    dummy_clf = train_dummy_classifier(X_train_tfidf, y_train, strategy='most_frequent')

    # Evaluate Logistic Regression
    log_reg_metrics = evaluate_model(log_reg_clf, X_test_tfidf, y_test, model_name="Logistic Regression")

    # Evaluate Dummy Classifier
    dummy_metrics = evaluate_model(dummy_clf, X_test_tfidf, y_test, model_name="Dummy Classifier")

    # Save models and vectorizer
    save_model(log_reg_clf, vectorizer, LOG_REG_MODEL_PATH, VECTORIZER_PATH)
    save_model(dummy_clf, vectorizer, DUMMY_MODEL_PATH, VECTORIZER_PATH)

    # Additional Step: Ignore 'Other' in Evaluation
    # Reload the test data
    test_sentences = X_test
    test_labels = y_test
    test_predictions_log_reg = log_reg_clf.predict(X_test_tfidf)
    test_predictions_dummy = dummy_clf.predict(X_test_tfidf)

    # Filter out 'Other' instances
    filtered_sentences, filtered_labels = filter_out_other(test_sentences, test_labels, other_label="Other")
    
    # Corresponding predictions
    filtered_predictions_log_reg = []
    filtered_predictions_dummy = []
    for sentence, true_label in zip(test_sentences, test_labels):
        if true_label != "Other":
            filtered_predictions_log_reg.append(log_reg_clf.predict(vectorizer.transform([sentence]))[0])
            filtered_predictions_dummy.append(dummy_clf.predict(vectorizer.transform([sentence]))[0])

    # Evaluate Logistic Regression without 'Other'
    print("\n=== Logistic Regression Evaluation (Excluding 'Other') ===")
    accuracy = accuracy_score(filtered_labels, filtered_predictions_log_reg)
    report = classification_report(filtered_labels, filtered_predictions_log_reg, zero_division=0)
    cm = confusion_matrix(filtered_labels, filtered_predictions_log_reg)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Evaluate Dummy Classifier without 'Other'
    print("\n=== Dummy Classifier Evaluation (Excluding 'Other') ===")
    accuracy_dummy = accuracy_score(filtered_labels, filtered_predictions_dummy)
    report_dummy = classification_report(filtered_labels, filtered_predictions_dummy, zero_division=0)
    cm_dummy = confusion_matrix(filtered_labels, filtered_predictions_dummy)
    
    print(f"Accuracy: {accuracy_dummy:.4f}")
    print("Classification Report:")
    print(report_dummy)
    print("Confusion Matrix:")
    print(cm_dummy)

if __name__ == "__main__":
    main()