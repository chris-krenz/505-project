import json
import re

# File paths
INPUT_FILE = "synthetic_biology_preprocessed.json"
OUTPUT_LABELS_FILE = "labels.json"

# Define your keywords and corresponding labels
# You can customize the labels as needed
KEYWORD_LABEL_MAP = {  # TODO: add multiple keywords per service (e.g. 'golden gate' and 'modular cloning' both map to the same service)
    "gibson assembly": "Gibson Assembly",
    "modular cloning": "Modular Cloning",
    "restriction digestion": "Restriction Digestion",
    "restriction ligation": "Restriction Ligation",
    "pcr reaction": "PCR Reaction",
    "colony pcr": "Colony PCR",
    "temperature gradient test": "Temperature Gradient Test",
    "pcr cleanup": "PCR Cleanup",
    "gel electrophoresis": "Gel Electrophoresis",
    "agarose gel extraction": "Agarose Gel Extraction",
    "concentrate dna": "Concentrate DNA",
    "ethanol precipitation": "Ethanol Precipitation",
    "dna extraction": "DNA Extraction",
    "plasmid miniprep": "Plasmid Miniprep",
    "glycerol stock": "Glycerol Stock",
    "plasmid midiprep": "Plasmid Midiprep",
    "plasmid maxiprep": "Plasmid Maxiprep",
    "sample to sequencing": "Sample to Sequencing",
    "rehydrate dna": "Rehydrate DNA",
    "order dna fragments": "Order DNA Fragments",
    "design and order primers": "Design and Order Primers",
    "spectrophotometric assay": "Spectrophotometric Assay",
    "qpcr assay": "qPCR Assay",
    "next-generation sequencing": "Next-Generation Sequencing",
    "nextseq 2000": "NextSeq 2000",
    "opentrons liquid handler": "Opentrons Liquid Handler",
    "hamilton liquid handler": "Hamilton Liquid Handler",
    "cell culture induction": "Cell Culture Induction",
    "cell lysate production": "Cell Lysate Production",
    "protein purification": "Protein Purification",
    "cell transformation": "Cell Transformation",
    "overnight inoculum": "Overnight Inoculum",
    "e. coli lb growth": "E. coli LB Growth",
    "e. coli m9 growth": "E. coli M9 Growth",
    "e. coli x agar plate": "E. coli X Agar Plate",
    "plasmid storage": "Plasmid Storage",
    "glycerol stock": "Glycerol Stock",
    "lb agar plate": "LB Agar Plate",
}

# Precompile regex patterns for efficiency and case-insensitivity
KEYWORD_PATTERNS = { 
    keyword: re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) 
    for keyword in KEYWORD_LABEL_MAP.keys()
}

def load_sentences(filepath):
    """Load preprocessed sentences from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences.")
        return sentences
    except Exception as e:
        print(f"Error loading sentences: {e}")
        return []

def assign_labels(sentences, keyword_patterns, label_map):
    """Assign labels to sentences based on keyword presence."""
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
    """Save the labels to a JSON file."""
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
