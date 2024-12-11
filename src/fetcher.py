# fetcher.py
import os
import time
import json
from dotenv import load_dotenv

from Bio import Entrez

from utils import logger
from config import ROOT_DIR

def load_keyword_label_map(filepath):
    """
    Load the keyword-label mapping from a JSON file.
    Returns a dictionary mapping keywords to labels.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            keyword_label_map = json.load(file)
        print(f"Loaded {len(keyword_label_map)} keyword-label pairs from {filepath}.")
        return keyword_label_map
    except Exception as e:
        print(f"Error loading keyword-label map from {filepath}: {e}")
        exit(1)

load_dotenv()

Entrez.email = os.getenv("PUBMED_EMAIL")

max_abstracts_per_keyword = 250

output_file = os.path.join(ROOT_DIR, "data", "synthetic_biology_abstracts.json")

keyword_label_map_path = os.path.join(ROOT_DIR, "src", "KEYWORD_LABEL_MAP.json")


def search_pubmed(keyword, max_results):
    query = f'"{keyword}"[Abstract]'
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        id_list = record.get("IdList", [])
        print(f"Keyword '{keyword}': Found {len(id_list)} articles.")
        return id_list
    except Exception as e:
        print(f"An error occurred during PubMed search for keyword '{keyword}': {e}")
        return []

def fetch_abstracts(pubmed_id_list):
    abstracts = []
    batch_size = 100
    
    for start in range(0, len(pubmed_id_list), batch_size):
        end = min(start + batch_size, len(pubmed_id_list))
        batch_ids = pubmed_id_list[start:end]
        ids = ','.join(batch_ids)
        
        try:
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for article in records.get('PubmedArticle', []):
                try:
                    pubmed_id = article['MedlineCitation']['PMID']
                    abstract_text = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                    
                    if isinstance(abstract_text, list):
                        abstract = ' '.join(abstract_text)
                    else:
                        abstract = abstract_text
                        
                    abstracts.append({
                        "PubMedID": pubmed_id,
                        "Abstract": abstract
                    })
                except KeyError:
                    pubmed_id = article['MedlineCitation']['PMID']
                    print(f"PubMed ID {pubmed_id} has no abstract.")
                    continue
            
            print(f"Fetched abstracts {start + 1} to {end}.")
            time.sleep(0.34)
        except Exception as e:
            print(f"An error occurred while fetching abstracts for IDs {ids}: {e}")
            time.sleep(1)
    
    return abstracts


def save_to_json(data, filepath):
    """
    Saves the given data to a JSON file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} abstracts to {filepath}.")
    except Exception as e:
        print(f"An error occurred while saving to JSON: {e}")


def main():
    keyword_label_map = load_keyword_label_map(keyword_label_map_path)
    
    pubmed_id_to_labels = {}
    
    for keyword, label in keyword_label_map.items():
        pubmed_ids = search_pubmed(keyword, max_abstracts_per_keyword)
        for pubmed_id in pubmed_ids:
            if pubmed_id in pubmed_id_to_labels:
                if label not in pubmed_id_to_labels[pubmed_id]:
                    pubmed_id_to_labels[pubmed_id].append(label)
            else:
                pubmed_id_to_labels[pubmed_id] = [label]
        time.sleep(0.34)
    
    unique_pubmed_ids = list(pubmed_id_to_labels.keys())
    print(f"Total unique PubMed IDs collected: {len(unique_pubmed_ids)}")
    
    abstracts = fetch_abstracts(unique_pubmed_ids)
    
    final_abstracts = []
    pubmed_id_set = set()
    for abstract in abstracts:
        pubmed_id = abstract["PubMedID"]
        if pubmed_id in pubmed_id_set:
            continue  # avoid duplicates
        labels = pubmed_id_to_labels.get(pubmed_id, [])
        final_abstracts.append({
            "PubMedID": pubmed_id,
            "Abstract": abstract["Abstract"],
            "Labels": labels
        })
        pubmed_id_set.add(pubmed_id)
    
    print(f"Total abstracts after removing duplicates: {len(final_abstracts)}")
    
    save_to_json(final_abstracts, output_file)

if __name__ == "__main__":
    main()
    