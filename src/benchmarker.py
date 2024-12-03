import os
import json
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from config import ROOT_DIR


def load_data(preprocessed_file, labels_file):
    """
    Load preprocessed sentences and their corresponding labels from JSON files.
    
    Args:
        preprocessed_file (str): Path to the JSON file containing preprocessed sentences.
        labels_file (str): Path to the JSON file containing labels.
        
    Returns:
        sentences (list): List of preprocessed sentences.
        labels (list): List of labels corresponding to each sentence.
    """
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
    
    # Ensure that the number of sentences matches the number of labels
    if len(sentences) != len(labels):
        print("Mismatch between number of sentences and labels.")
        return [], []
    
    # If labels are lists (multi-label), convert to single labels (for simplicity)
    # Here, we assume each sentence has at least one label and take the first one
    # Modify this part if you have multi-label data and want to handle it differently
    single_labels = [label_list[0] if label_list else "Other" for label_list in labels]
    
    return sentences, single_labels

def vectorize_text(X_train, X_test, max_features=1000):
    """
    Convert text data into TF-IDF feature vectors.
    
    Args:
        X_train (list): Training sentences.
        X_test (list): Testing sentences.
        max_features (int): Maximum number of features for TF-IDF.
        
    Returns:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        X_train_tfidf (sparse matrix): TF-IDF features for training data.
        X_test_tfidf (sparse matrix): TF-IDF features for testing data.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vectorized text with TF-IDF. Number of features: {max_features}")
    return vectorizer, X_train_tfidf, X_test_tfidf

def analyze_label_distribution(labels):
    """
    Analyze and print the distribution of labels.

    Args:
        labels (list): List of labels.

    Returns:
        problematic_classes (list): List of classes with fewer than 2 instances.
    """
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)
    
    # Identify classes with fewer than 2 instances
    problematic_classes = [label for label, count in label_counts.items() if count < 2]
    print(f"Classes with fewer than 2 instances: {problematic_classes}")
    
    return problematic_classes

def merge_rare_classes(labels, problematic_classes, merge_into="Other"):
    """
    Merge rare classes into a specified label.

    Args:
        labels (list): Original list of labels.
        problematic_classes (list): Classes to be merged.
        merge_into (str): The label to merge rare classes into.

    Returns:
        merged_labels (list): Updated list of labels.
    """
    merged_labels = [label if label not in problematic_classes else merge_into for label in labels]
    print(f"Merged {len(problematic_classes)} rare classes into '{merge_into}'.")
    return merged_labels

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    
    Args:
        X_train (sparse matrix): TF-IDF features for training data.
        y_train (list): Training labels.
        
    Returns:
        clf (LogisticRegression): Trained Logistic Regression model.
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    print("Trained Logistic Regression classifier.")
    return clf

def train_dummy_classifier(X_train, y_train, strategy='most_frequent'):
    """
    Train a Dummy classifier as a baseline.
    
    Args:
        X_train (sparse matrix): TF-IDF features for training data.
        y_train (list): Training labels.
        strategy (str): Strategy for the Dummy classifier ('most_frequent', 'stratified', etc.).
        
    Returns:
        dummy_clf (DummyClassifier): Trained Dummy classifier.
    """
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    dummy_clf.fit(X_train, y_train)
    print(f"Trained Dummy classifier with strategy='{strategy}'.")
    return dummy_clf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on the test set and print metrics.
    
    Args:
        model: Trained classifier.
        X_test (sparse matrix): TF-IDF features for testing data.
        y_test (list): True labels for testing data.
        model_name (str): Name of the model for reporting.
        
    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
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
    
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }
    return metrics

def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    Save the trained model and vectorizer to disk.
    
    Args:
        model: Trained classifier.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        model_path (str): Path to save the model.
        vectorizer_path (str): Path to save the vectorizer.
    """
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

def main():
    # File paths
    PREPROCESSED_FILE  = os.path.join(ROOT_DIR, "data", "synthetic_biology_preprocessed.json")
    LABELS_FILE        = os.path.join(ROOT_DIR, "data", "labels.json")
    LOG_REG_MODEL_PATH = os.path.join(ROOT_DIR, "data", "logistic_regression_model.pkl")
    DUMMY_MODEL_PATH   = os.path.join(ROOT_DIR, "data", "dummy_classifier.pkl")
    VECTORIZER_PATH    = os.path.join(ROOT_DIR, "data", "tfidf_vectorizer.pkl")
    
    # Load data
    sentences, labels = load_data(PREPROCESSED_FILE, LABELS_FILE)
    if not sentences or not labels:
        print("Data loading failed. Exiting.")
        return
    
    # Analyze label distribution
    problematic_classes = analyze_label_distribution(labels)
    
    # Merge rare classes into 'Other'
    if problematic_classes:
        labels = merge_rare_classes(labels, problematic_classes, merge_into="Other")
    
    # Re-analyze label distribution after merging
    analyze_label_distribution(labels)
    
    # Split data into training and testing sets with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, labels, test_size=0.2, random_state=42, stratify=labels
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
    
    # Save the metrics to a JSON file for later analysis
    metrics_summary = {
        "Logistic Regression": log_reg_metrics,
        "Dummy Classifier": dummy_metrics
    }
    try:
        with open("model_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)
        print("Saved evaluation metrics to model_metrics.json.")
    except Exception as e:
        print(f"Error saving metrics to model_metrics.json: {e}")


if __name__ == "__main__":
    main()
