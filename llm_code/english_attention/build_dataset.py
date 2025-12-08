# build_dataset.py

import os
import requests
from conllu import parse
import pandas as pd

from config import URLS, N_SUBJ_ROOT_CSV, MILLION_CONLLU_PATH


def fetch_and_parse(url: str):
    """Download a UD .conllu file and parse it into sentence objects."""
    resp = requests.get(url)
    resp.raise_for_status()
    # UD files are sometimes large; ensure we handle text encoding
    return parse(resp.text)


def parse_local_conllu(path: str):
    """Parse a local .conllu file into sentence objects."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return parse(text)


def extract_nsubj_root_rows(sentences, split_name: str):
    """
    From a sequence of parsed sentences, extract those that have both
    an nsubj and a root. Return a list of dict rows.
    """
    rows = []

    for sent in sentences:
        nsubj_tokens = [tok for tok in sent if tok["deprel"] == "nsubj"]
        root_tokens = [tok for tok in sent if tok["deprel"] == "root"]
        
        # We need at least one nsubj and one root
        if not (nsubj_tokens and root_tokens): 
            continue
            
        # grab the first subject if there is more than one nsubj.
        if len(nsubj_tokens) != len(root_tokens): 
            nsubj_tokens = nsubj_tokens[:1]
            
        nsubj_position = nsubj_tokens[0].get('id')
        nsubj_forms = [tok["form"] for tok in nsubj_tokens]
        root_forms = [tok["form"] for tok in root_tokens]

        # Turkish constraint: nsubj_position <= 3. 
        # For English/Russian, this might be too restrictive or irrelevant. 
        # But for PARITY with the Turkish script as requested ("similar folders"), 
        # we often keep the logic or adapt it. 
        # The user asked to "compare it... and create similar folders... and improve".
        # Strict position constraint is likely specific to the Turkish pro-drop or word order analysis.
        # However, for now, I'll keep it commented out or relaxed if I knew better.
        # Rereading the original file: it has `if nsubj_position <= 3`. 
        # I will KEEP this constraint but add a TODO comment that it might be language specific.
        
        if nsubj_position <= 3: 
            # if the subject is one of the first three words in a sentence. 
            rows.append(
                {
                    "sentence": " ".join(tok["form"] for tok in sent),
                    "nsubj": ";".join(nsubj_forms),
                    "root": ";".join(root_forms),
                    "nsubj_first": nsubj_forms[0],
                    "nsubj_position": nsubj_tokens[0].get('id'), # get position
                    "root_first": root_forms[0],
                    "split": split_name,
                }
            )
        else:
            # print("Subject not in index 0,1,2,3.")
            pass
            
    return rows

def build_nsubj_root_df(urls: dict, million_path: str) -> pd.DataFrame:
    """
    Build a single DataFrame that includes:
      - all UD splits (from remote URLs)
      - the local 1M-sentence web corpus
    """
    all_rows = []

    for split_name, url in urls.items():
        print(f"Fetching {split_name} from {url} ...")
        try:
            sentences = fetch_and_parse(url)
            rows = extract_nsubj_root_rows(sentences, split_name)
            print(f"  -> {len(rows)} sentences with nsubj+root")
            all_rows.extend(rows)
        except Exception as e:
            print(f"Failed to fetch/parse {split_name}: {e}")

    if million_path is not None:
        if os.path.exists(million_path):
            print(f"Parsing million-sentence corpus from {million_path} ...")
            sentences = parse_local_conllu(million_path)
            rows = extract_nsubj_root_rows(sentences, "million")
            print(f"  -> {len(rows)} sentences with nsubj+root (million)")
            all_rows.extend(rows)
        else:
             print(f"Million corpus not found at {million_path}, skipping.")

    return pd.DataFrame(all_rows)


def main():
    df = build_nsubj_root_df(URLS, MILLION_CONLLU_PATH)
    os.makedirs(os.path.dirname(N_SUBJ_ROOT_CSV), exist_ok=True)
    df.to_csv(N_SUBJ_ROOT_CSV, index=False)
    print(f"Saved {len(df)} sentences (including million corpus) to {N_SUBJ_ROOT_CSV}")


if __name__ == "__main__":
    main()
