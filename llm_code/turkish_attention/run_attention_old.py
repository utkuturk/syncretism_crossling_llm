# run_attention.py

import os
import argparse
import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    GPT2Tokenizer,
    GPT2Model,
)

# Allow both "package" and "script" usage
try:
    from .config import (
        BERT_MODEL_NAME,
        GPT2_MODEL_NAME,
        N_SUBJ_ROOT_CSV,
        BERT_ATTENTION_CSV,
        GPT2_ATTENTION_CSV,
    )
except ImportError:  # running as plain script in the same folder
    from config import (
        BERT_MODEL_NAME,
        GPT2_MODEL_NAME,
        N_SUBJ_ROOT_CSV,
        BERT_ATTENTION_CSV,
        GPT2_ATTENTION_CSV,
    )


def load_dataset(path: str = N_SUBJ_ROOT_CSV) -> pd.DataFrame:
    """Load the sentence / nsubj / root dataset."""
    return pd.read_csv(path)


# ---------- helper functions for subword alignment ----------

def find_wordpiece_span_bert(word: str, tokenizer, input_ids):
    """
    Find the start/end token indices for `word` inside `input_ids`
    for BERT (WordPiece). Returns (start, end) or None.
    """
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    if not word_ids:
        return None

    sent_ids = input_ids[0].tolist()
    max_start = len(sent_ids) - len(word_ids)

    for start in range(max_start + 1):
        if sent_ids[start:start + len(word_ids)] == word_ids:
            end = start + len(word_ids) - 1
            return start, end

    return None


def find_subtoken_span_gpt2(word: str, tokenizer, input_ids):
    """
    Find the start/end token indices for `word` inside `input_ids`
    for GPT-2 (byte-level BPE). Returns (start, end) or None.
    Tries both ' word' and 'word' patterns.
    """
    sent_ids = input_ids[0].tolist()

    patterns = [
        tokenizer.encode(" " + word, add_special_tokens=False),
        tokenizer.encode(word, add_special_tokens=False),
    ]

    for pattern in patterns:
        if not pattern:
            continue

        max_start = len(sent_ids) - len(pattern)
        for start in range(max_start + 1):
            if sent_ids[start:start + len(pattern)] == pattern:
                end = start + len(pattern) - 1
                return start, end

    return None


# ---------- BERT ----------

def compute_bert_attention(df: pd.DataFrame, log_every: int = 500) -> pd.DataFrame:
    """
    For each sentence, find the strongest attention from root to nsubj
    across all layers/heads in Turkish BERT.
    Uses subword ID sequence matching instead of exact-token matching.
    Skips sentences longer than the model's max sequence length.
    """
    print("[BERT] Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME, output_attentions=True)
    model.eval()

    max_len = model.config.max_position_embeddings
    total_rows = len(df)
    print(f"[BERT] Rows in this chunk: {total_rows} (max_len={max_len})")

    results = []
    skipped_no_span = 0
    skipped_order = 0
    skipped_too_long = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["sentence"]
        nsubj = row["nsubj_first"]
        root = row["root_first"]

        # Encode once with special tokens to check length and use for spans/model
        inputs = tokenizer(sentence, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        if seq_len > max_len:
            skipped_too_long += 1
            if i % log_every == 0:
                print(
                    f"[BERT] {i}/{total_rows} rows | kept: {len(results)} | "
                    f"skipped (too long): {skipped_too_long} | "
                    f"skipped (no span): {skipped_no_span} | "
                    f"skipped (order): {skipped_order}",
                    flush=True,
                )
            continue

        nsubj_span = find_wordpiece_span_bert(nsubj, tokenizer, inputs["input_ids"])
        root_span = find_wordpiece_span_bert(root, tokenizer, inputs["input_ids"])

        if nsubj_span is None or root_span is None:
            skipped_no_span += 1
            if i % log_every == 0:
                print(
                    f"[BERT] {i}/{total_rows} rows | kept: {len(results)} | "
                    f"skipped (too long): {skipped_too_long} | "
                    f"skipped (no span): {skipped_no_span} | "
                    f"skipped (order): {skipped_order}",
                    flush=True,
                )
            continue

        nsubj_idx = nsubj_span[0]
        root_idx = root_span[0]

        # keep cases where root comes after subject
        if root_idx < nsubj_idx:
            skipped_order += 1
            if i % log_every == 0:
                print(
                    f"[BERT] {i}/{total_rows} rows | kept: {len(results)} | "
                    f"skipped (too long): {skipped_too_long} | "
                    f"skipped (no span): {skipped_no_span} | "
                    f"skipped (order): {skipped_order}",
                    flush=True,
                )
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions  # list of (batch, heads, seq, seq)
        attn_tensor = torch.stack(attentions).squeeze(1)  # (layers, heads, seq, seq)

        attention_scores = attn_tensor[:, :, root_idx, nsubj_idx]  # (layers, heads)

        flat_idx = torch.argmax(attention_scores)
        num_heads = attention_scores.shape[1]

        best_layer = int(flat_idx // num_heads)
        best_head = int(flat_idx % num_heads)
        max_value = float(attention_scores[best_layer, best_head].item())

        results.append(
            {
                "sentence": sentence,
                "nsubj": nsubj,
                "root": root,
                "nsubj_idx": int(nsubj_idx),
                "root_idx": int(root_idx),
                "best_layer": best_layer,
                "best_head": best_head,
                "attention_value": max_value,
            }
        )

        if i % log_every == 0:
            print(
                f"[BERT] {i}/{total_rows} rows | kept: {len(results)} | "
                f"skipped (too long): {skipped_too_long} | "
                f"skipped (no span): {skipped_no_span} | "
                f"skipped (order): {skipped_order}",
                flush=True,
            )

    print(
        f"[BERT] Done for this chunk. Rows: {total_rows} | "
        f"kept: {len(results)} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span} | "
        f"skipped (order): {skipped_order}"
    )

    return pd.DataFrame(results)


# ---------- GPT-2 ----------

def compute_gpt2_attention(df: pd.DataFrame, log_every: int = 500) -> pd.DataFrame:
    """
    For each sentence, find the strongest attention from root to nsubj
    across all layers/heads in Turkish GPT-2.
    Uses subtoken ID sequence matching instead of simple substring matching.
    Skips sentences longer than the model's max sequence length.
    """
    print("[GPT-2] Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(GPT2_MODEL_NAME, output_attentions=True)
    model.eval()

    # GPT-2 uses n_positions as its max context length
    max_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    total_rows = len(df)
    print(f"[GPT-2] Rows in this chunk: {total_rows} (max_len={max_len})")

    results = []
    skipped_no_span = 0
    skipped_order = 0
    skipped_too_long = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["sentence"]
        nsubj = row["nsubj_first"]
        root = row["root_first"]

        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        seq_len = inputs["input_ids"].shape[1]

        if max_len is not None and seq_len > max_len:
            skipped_too_long += 1
            if i % log_every == 0:
                print(
                    f"[GPT-2] {i}/{total_rows} rows | kept: {len(results)} | "
                    f"skipped (too long): {skipped_too_long} | "
                    f"skipped (no span): {skipped_no_span} | "
                    f"skipped (order): {skipped_order}",
                    flush=True,
                )
            continue

        root_span = find_subtoken_span_gpt2(root, tokenizer, inputs["input_ids"])
        nsubj_span = find_subtoken_span_gpt2(nsubj, tokenizer, inputs["input_ids"])

        if root_span is None or nsubj_span is None:
            skipped_no_span += 1
            if i % log_every == 0:
                print(
                    f"[GPT-2] {i}/{total_rows} rows | kept: {len(results)} | "
                    f"skipped (too long): {skipped_too_long} | "
                    f"skipped (no span): {skipped_no_span} | "
                    f"skipped (order): {skipped_order}",
                    flush=True,
                )
            continue

        root_idx = root_span[0]
        nsubj_idx = nsubj_span[0]

        # only keep subject-before-verb direction
        if nsubj_idx >= root_idx:
            skipped_order += 1
            if i % log_every == 0:
                print(
                    f"[GPT-2] {i}/{total_rows} rows | kept: {len(results)} | "
                    f"skipped (too long): {skipped_too_long} | "
                    f"skipped (no span): {skipped_no_span} | "
                    f"skipped (order): {skipped_order}",
                    flush=True,
                )
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions  # list of (batch, heads, seq, seq)
        attn_tensor = torch.stack(attentions).squeeze(1)  # (layers, heads, seq, seq)

        attention_scores = attn_tensor[:, :, root_idx, nsubj_idx]
        flat_idx = torch.argmax(attention_scores)
        num_heads = attention_scores.shape[1]

        best_layer = int(flat_idx // num_heads)
        best_head = int(flat_idx % num_heads)
        max_value = float(attention_scores[best_layer, best_head].item())

        results.append(
            {
                "sentence": sentence,
                "nsubj": nsubj,
                "root": root,
                "nsubj_idx": int(nsubj_idx),
                "root_idx": int(root_idx),
                "best_layer": best_layer,
                "best_head": best_head,
                "attention_value": max_value,
            }
        )

        if i % log_every == 0:
            print(
                f"[GPT-2] {i}/{total_rows} rows | kept: {len(results)} | "
                f"skipped (too long): {skipped_too_long} | "
                f"skipped (no span): {skipped_no_span} | "
                f"skipped (order): {skipped_order}",
                flush=True,
            )

    print(
        f"[GPT-2] Done for this chunk. Rows: {total_rows} | "
        f"kept: {len(results)} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span} | "
        f"skipped (order): {skipped_order}"
    )

    return pd.DataFrame(results)


# ---------- chunked processing ----------

def process_chunk(df: pd.DataFrame, chunk_start: int, chunk_end: int, chunk_id: int, log_every: int = 1000):
    """
    Process a single chunk [chunk_start:chunk_end) and save one CSV per model.
    """
    df_chunk = df.iloc[chunk_start:chunk_end].reset_index(drop=True)
    print(
        f"\n=== Processing chunk {chunk_id} "
        f"(rows {chunk_start}–{chunk_end}, {len(df_chunk)} rows) ==="
    )

    os.makedirs("results", exist_ok=True)

    # BERT
    bert_df = compute_bert_attention(df_chunk, log_every=log_every)
    bert_base = os.path.splitext(BERT_ATTENTION_CSV)[0]
    bert_path = f"{bert_base}_chunk_{chunk_id:04d}.csv"
    bert_df.to_csv(bert_path, index=False)
    print(f"[BERT] Saved chunk {chunk_id} ({len(bert_df)} rows) to {bert_path}")

    # GPT-2
    gpt2_df = compute_gpt2_attention(df_chunk, log_every=log_every)
    gpt2_base = os.path.splitext(GPT2_ATTENTION_CSV)[0]
    gpt2_path = f"{gpt2_base}_chunk_{chunk_id:04d}.csv"
    gpt2_df.to_csv(gpt2_path, index=False)
    print(f"[GPT-2] Saved chunk {chunk_id} ({len(gpt2_df)} rows) to {gpt2_path}")


def main(chunk_size: int = 100000, skip_chunks: int = 0, num_chunks: int = 1, log_every: int = 1000):
    df = load_dataset()
    total_rows = len(df)
    print(f"[MAIN] Total rows in dataset: {total_rows}")

    start_row = skip_chunks * chunk_size
    if start_row >= total_rows:
        print(f"[MAIN] skip_chunks={skip_chunks} (start_row={start_row}) is beyond dataset size.")
        return

    for k in range(num_chunks):
        chunk_id = skip_chunks + k
        chunk_start = start_row + k * chunk_size
        if chunk_start >= total_rows:
            print(f"[MAIN] Reached end of dataset at chunk {chunk_id}. Stopping.")
            break

        chunk_end = min(chunk_start + chunk_size, total_rows)
        process_chunk(df, chunk_start, chunk_end, chunk_id, log_every=log_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BERT/GPT-2 attention in chunks.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows per chunk (default: 100000).",
    )
    parser.add_argument(
        "--skip-chunks",
        type=int,
        default=0,
        help="How many chunks to skip before starting (default: 0).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="How many chunks to process in this run (default: 1).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N rows within each chunk (default: 1000).",
    )

    args = parser.parse_args()

    main(
        chunk_size=args.chunk_size,
        skip_chunks=args.skip_chunks,
        num_chunks=args.num_chunks,
        log_every=args.log_every,
    )
