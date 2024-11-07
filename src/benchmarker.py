import os
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

def vectorize_text(X_train, X_test, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vectorized text with TF-IDF. Number of features: {max_features}")
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    print("Trained Logistic Regression classifier.")
    return clf

def train_dummy_classifier(X_train, y_train, strategy='most_frequent'):
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    dummy_clf.fit(X_train, y_train)
    print(f"Trained Dummy classifier with strategy='{strategy}'.")
    return dummy_clf

def evaluate_model(model, X_test, y_test, model_name="Model"):
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
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples.")
    
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
    
    # Save the metrics to a JSON or text file for later analysis
    metrics_summary = {
        "Logistic Regression": log_reg_metrics,
        "Dummy Classifier": dummy_metrics
    }
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Saved evaluation metrics to model_metrics.json.")


if __name__ == "__main__":
    main()
