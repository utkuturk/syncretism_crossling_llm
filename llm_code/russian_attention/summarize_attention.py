# summarize_attention.py

import os
import glob
import pandas as pd

from config import BERT_ATTENTION_CSV, GPT2_ATTENTION_CSV

DATA_DIR = "../../data"
RESULTS_DIR = "results"

def load_all_chunks(chunk_pattern: str, model_name: str) -> pd.DataFrame:
    """
    Load and concatenate all chunk CSVs matching the pattern.
    """
    files = sorted(glob.glob(chunk_pattern))
    if not files:
        print(f"[{model_name}] No files found for pattern: {chunk_pattern}. Checking for full file?")
        return pd.DataFrame()

    print(f"[{model_name}] Found {len(files)} chunk files.")
    dfs = []
    for f in files:
        print(f"[{model_name}] Loading {f} ...")
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)
    print(f"[{model_name}] Total rows after concatenation: {len(df)}")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table:

      best_layer, best_head,
      total_weighted_attention, count, accuracy (% of sentences)
    """
    if df.empty:
        return pd.DataFrame()
        
    total_sentences = len(df)

    weighted_scores = (
        df.groupby(["best_layer", "best_head"])["attention_value"]
        .sum()
        .reset_index()
        .rename(columns={"attention_value": "total_weighted_attention"})
    )

    frequency = (
        df.groupby(["best_layer", "best_head"])
        .size()
        .reset_index(name="count")
    )

    summary = weighted_scores.merge(frequency, on=["best_layer", "best_head"])
    summary["accuracy"] = 100.0 * summary["count"] / total_sentences

    return summary


def print_top5(summary_df: pd.DataFrame, model_name: str):
    if summary_df.empty:
        print(f"[{model_name}] Summary empty.")
        return

    print(f"\n=== {model_name} top 5 by total weighted attention ===")
    print(
        summary_df.sort_values("total_weighted_attention", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    print(f"\n=== {model_name} top 5 by accuracy (% sentences best) ===")
    print(
        summary_df.sort_values("accuracy", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Construct patterns based on config paths
    bert_base = os.path.splitext(BERT_ATTENTION_CSV)[0]
    gpt2_base = os.path.splitext(GPT2_ATTENTION_CSV)[0]

    bert_chunk_pattern = f"{bert_base}_chunk_*.csv"
    gpt2_chunk_pattern = f"{gpt2_base}_chunk_*.csv"

    # -------- GPT-2 --------
    print("### Summarizing GPT-2 attention ###")
    gpt2_df = load_all_chunks(gpt2_chunk_pattern, "GPT-2")
    if not gpt2_df.empty:
        gpt2_summary = build_summary(gpt2_df)
        print_top5(gpt2_summary, "GPT-2")

        gpt2_summary_sorted = gpt2_summary.sort_values("accuracy", ascending=False)
        gpt2_heads_csv = os.path.join(RESULTS_DIR, "gpt2_heads_by_accuracy.csv")
        gpt2_summary_sorted.to_csv(gpt2_heads_csv, index=False)
        print(f"[GPT-2] Head/layer summary sorted by accuracy saved to: {gpt2_heads_csv}")

    # -------- BERT --------
    print("\n\n### Summarizing BERT attention ###")
    bert_df = load_all_chunks(bert_chunk_pattern, "BERT")
    if not bert_df.empty:
        bert_summary = build_summary(bert_df)
        print_top5(bert_summary, "BERT")

        bert_summary_sorted = bert_summary.sort_values("accuracy", ascending=False)
        bert_heads_csv = os.path.join(RESULTS_DIR, "bert_heads_by_accuracy.csv")
        bert_summary_sorted.to_csv(bert_heads_csv, index=False)
        print(f"[BERT] Head/layer summary sorted by accuracy saved to: {bert_heads_csv}")


if __name__ == "__main__":
    main()
