import os
import time
import json
import pickle
from dotenv import load_dotenv

import openai

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from config import ROOT_DIR

load_dotenv()

with open('KEYWORD_LABEL_MAP.json', 'r', encoding='utf-8') as file:
    KEYWORD_LABEL_MAP = json.load(file)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

PREPROCESSED_FILE        = os.path.join(ROOT_DIR, "data", "synthetic_biology_preprocessed.json")
GROUND_TRUTH_LABELS_FILE = os.path.join(ROOT_DIR, "data", "labels.json")
GPT4_LABELS_FILE         = os.path.join(ROOT_DIR, "data", "gpt4_labels.json")

import re
KEYWORD_PATTERNS = { 
    keyword: re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) 
    for keyword in KEYWORD_LABEL_MAP.keys()
}


def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {filepath}.")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def save_json(data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} entries to {filepath}.")
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")

def assign_labels_with_keywords(sentences, keyword_patterns, label_map):
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
            print(f"Processed {idx + 1}/{len(sentences)} sentences for keyword labeling.")
    return labels

def generate_gpt4_labels(sentences, keyword_list, batch_size=10, sleep_time=1):
    gpt4_labels = []
    total_sentences = len(sentences)
    for i in range(0, total_sentences, batch_size):
        batch_sentences = sentences[i:i+batch_size]
        prompt = (
            "You are an assistant that labels sentences based on the presence of specific keywords.\n"
            f"Keywords and their corresponding labels:\n"
        )
        for keyword, label in KEYWORD_LABEL_MAP.items():
            prompt += f"- {keyword}: {label}\n"
        prompt += "\nLabel each of the following sentences with the appropriate label(s). If none of the keywords are present, label the sentence as 'Other'.\n\n"
        for idx, sentence in enumerate(batch_sentences, 1):
            prompt += f"Sentence {i + idx}: \"{sentence}\"\nLabel {i + idx}: "
        prompt += "\nPlease provide only the labels, separated by commas if multiple."

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for labeling sentences based on provided keywords."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # Set temperature to 0 for deterministic output
            )
            labels_text = response.choices[0].message['content'].strip()
            # Split the response into individual labels
            batch_labels = [label.strip() for label in labels_text.split('\n') if label.strip()]
            # Handle cases where GPT-4 returns labels separated by commas
            final_batch_labels = []
            for label in batch_labels:
                # If multiple labels are present, split by comma
                split_labels = [lbl.strip() for lbl in label.split(',')]
                final_batch_labels.append(split_labels)
            # Append to the main list
            gpt4_labels.extend(final_batch_labels)
            print(f"Processed batch {i // batch_size +1} / {((total_sentences -1) // batch_size) +1}")
            time.sleep(sleep_time)  # To respect rate limits
        except Exception as e:
            print(f"Error during GPT-4 labeling at batch starting index {i}: {e}")
            # Optionally, retry or skip
            for _ in batch_sentences:
                gpt4_labels.append(["Other"])  # Default label in case of error
    return gpt4_labels

def flatten_labels(labels):
    return [", ".join(label_list) for label_list in labels]

def evaluate_predictions(y_true, y_pred, average='macro'):
    print("\n=== GPT-4 Evaluation ===")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    # Load preprocessed sentences and ground truth labels
    sentences = load_json(PREPROCESSED_FILE)
    ground_truth_labels = load_json(GROUND_TRUTH_LABELS_FILE)

    if not sentences or not ground_truth_labels:
        print("Failed to load sentences or labels. Exiting.")
        return

    if len(sentences) != len(ground_truth_labels):
        print("Mismatch between number of sentences and labels. Exiting.")
        return

    # Assign keyword-based labels for comparison
    keyword_labels = assign_labels_with_keywords(sentences, KEYWORD_PATTERNS, KEYWORD_LABEL_MAP)
    save_json(keyword_labels, os.path.join(ROOT_DIR, "data", "keyword_labels.json"))  # Optional: Save keyword-based labels

    # Generate GPT-4 labels
    keyword_list = list(KEYWORD_LABEL_MAP.keys())
    gpt4_labels = generate_gpt4_labels(sentences, keyword_list, batch_size=10, sleep_time=1)
    save_json(gpt4_labels, GPT4_LABELS_FILE)

    # Prepare labels for evaluation
    # Assuming single-label classification (taking the first label)
    y_true = [label_list[0] for label_list in ground_truth_labels]
    y_gpt4 = [label_list[0] for label_list in gpt4_labels]

    # Evaluate GPT-4 predictions
    evaluate_predictions(y_true, y_gpt4)


if __name__ == "__main__":
    main()
