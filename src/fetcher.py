import os
import time
import json
from dotenv import load_dotenv

from Bio import Entrez

from utils import logger
from config import ROOT_DIR

with open('KEYWORD_LABEL_MAP.json', 'r', encoding='utf-8') as file:
    KEYWORDS = json.load(file).keys()

load_dotenv()


# TODO: VSCode prompted to remove unusual line terminators, such as PS, which I did...


Entrez.email = os.getenv("PUBMED_EMAIL")  # Replace with your actual email

search_query = ' OR '.join([f'"{keyword}"' for keyword in KEYWORDS])

max_abstracts = 10_000 

output_file = os.path.join(ROOT_DIR, "data", "synthetic_biology_abstracts.json")


def search_pubmed(query, max_results):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()
        print(f"Found {len(id_list)} articles.")
        return id_list
    except Exception as e:
        print(f"An error occurred during PubMed search: {e}")
        return []

# --------------- Fetch Abstracts ---------------

def fetch_abstracts(pubmed_ids):
    abstracts = []
    batch_size = 100  # Number of IDs to fetch per batch (adjust as needed)
    
    for start in range(0, len(pubmed_ids), batch_size):
        end = min(start + batch_size, len(pubmed_ids))
        batch_ids = pubmed_ids[start:end]
        ids = ','.join(batch_ids)
        
        try:
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for article in records['PubmedArticle']:
                try:
                    pubmed_id = article['MedlineCitation']['PMID']
                    abstract_text = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                    
                    # Some abstracts have multiple sections; join them if necessary
                    if isinstance(abstract_text, list):
                        abstract = ' '.join(abstract_text)
                    else:
                        abstract = abstract_text
                        
                    abstracts.append({
                        "PubMedID": pubmed_id,
                        "Abstract": abstract
                    })
                except KeyError:
                    # Handle articles without abstracts
                    print(f"PubMed ID {article['MedlineCitation']['PMID']} has no abstract.")
                    continue
            
            print(f"Fetched abstracts {start + 1} to {end}.")
            time.sleep(0.5)  # Respect NCBI rate limits
        except Exception as e:
            print(f"An error occurred while fetching abstracts for IDs {ids}: {e}")
            time.sleep(1)  # Wait before retrying or continuing
    
    return abstracts

# --------------- Save to JSON ---------------

def save_to_json(data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} abstracts to {filepath}.")
    except Exception as e:
        print(f"An error occurred while saving to JSON: {e}")

# --------------- Main Execution ---------------

def main():
    # Step 1: Search PubMed
    pubmed_ids = search_pubmed(search_query, max_abstracts)
    
    if not pubmed_ids:
        print("No PubMed IDs found. Exiting.")
        return
    
    # Step 2: Fetch Abstracts
    abstracts = fetch_abstracts(pubmed_ids)
    
    if not abstracts:
        print("No abstracts fetched. Exiting.")
        return
    
    # Step 3: Save to JSON
    save_to_json(abstracts, output_file)

if __name__ == "__main__":
    main()
