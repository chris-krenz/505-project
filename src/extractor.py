import os
import json

import nltk
from nltk.tokenize import sent_tokenize

from config import ROOT_DIR

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt_tab')  # TODO: Use 'punkt' instead

# TODO: add multiple keywords per service (e.g. 'golden gate' and 'modular cloning' both map to the same service)
with open('KEYWORD_LABEL_MAP.json', 'r', encoding='utf-8') as file:
    KEYWORDS = json.load(file).keys()

# File paths
INPUT_JSON  = os.path.join(ROOT_DIR, "data", "synthetic_biology_abstracts.json")
OUTPUT_JSON = os.path.join(ROOT_DIR, "data", "synthetic_biology_corpus.json")  # You can also use .txt if preferred


def load_abstracts(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            abstracts = json.load(file)
        print(f"Loaded {len(abstracts)} abstracts.")
        return abstracts
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def extract_sentences(abstracts, keywords):
    corpus = []
    for entry in abstracts:
        abstract = entry.get("Abstract", "")
        if not abstract:
            continue  # Skip if abstract is missing
        sentences = sent_tokenize(abstract)
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                corpus.append(sentence)
    print(f"Extracted {len(corpus)} relevant sentences.")
    return corpus

def save_corpus(corpus, filepath):
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(corpus, file, ensure_ascii=False, indent=2)
        elif filepath.endswith('.txt'):
            with open(filepath, 'w', encoding='utf-8') as file:
                for sentence in corpus:
                    file.write(sentence + '\n')
        else:
            print("Unsupported file format. Please use .json or .txt")
            return
        print(f"Saved {len(corpus)} sentences to {filepath}.")
    except Exception as e:
        print(f"Error saving corpus: {e}")

def main():
    abstracts = load_abstracts(INPUT_JSON)
    if not abstracts:
        print("No abstracts to process.")
        return
    corpus = extract_sentences(abstracts, KEYWORDS)
    if not corpus:
        print("No relevant sentences found.")
        return
    save_corpus(corpus, OUTPUT_JSON)

if __name__ == "__main__":
    main()
