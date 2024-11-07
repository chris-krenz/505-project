import os
import json
import re

from config import ROOT_DIR

with open('KEYWORD_LABEL_MAP.json', 'r', encoding='utf-8') as file:
    KEYWORD_LABEL_MAP = json.load(file)

# File paths
INPUT_FILE         = os.path.join(ROOT_DIR, "data", "synthetic_biology_preprocessed.json")
OUTPUT_LABELS_FILE = os.path.join(ROOT_DIR, "data", "labels.json")

# Define your keywords and corresponding labels
# You can customize the labels as needed

# Precompile regex patterns for efficiency and case-insensitivity
KEYWORD_PATTERNS = { 
    keyword: re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) 
    for keyword in KEYWORD_LABEL_MAP.keys()
}


def load_sentences(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences.")
        return sentences
    except Exception as e:
        print(f"Error loading sentences: {e}")
        return []

def assign_labels(sentences, keyword_patterns, label_map):
    labels = []
    for idx, sentence in enumerate(sentences):
        sentence_labels = []
        for keyword, pattern in keyword_patterns.items():
            if pattern.search(sentence):
                sentence_labels.append(label_map[keyword])
        if not sentence_labels:
            sentence_labels.append("Other")  # Label for sentences without keywords
        labels.append(sentence_labels)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(sentences):
            print(f"Processed {idx + 1}/{len(sentences)} sentences.")
    return labels

def save_labels(labels, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(labels)} labels to {filepath}.")
    except Exception as e:
        print(f"Error saving labels: {e}")

def main():
    # Load sentences
    sentences = load_sentences(INPUT_FILE)
    if not sentences:
        print("No sentences to process. Exiting.")
        return
    
    # Assign labels
    labels = assign_labels(sentences, KEYWORD_PATTERNS, KEYWORD_LABEL_MAP)
    
    # Save labels
    save_labels(labels, OUTPUT_LABELS_FILE)

if __name__ == "__main__":
    main()
