# entropy_run_chunks.py
#
# Compute attention entropy over selected heads for GPT-2 / BERT in chunks.
# Each chunk produces token-level entropy CSVs. No summarization here.

import os
import argparse
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel

try:
    from .config import (
        GPT2_MODEL_NAME,
        BERT_MODEL_NAME,
        N_SUBJ_ROOT_CSV,
    )
except ImportError:
    from config import (
        GPT2_MODEL_NAME,
        BERT_MODEL_NAME,
        N_SUBJ_ROOT_CSV,
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- head selection ----------

def select_heads_from_csv(
    csv_path: str,
    accuracy_threshold: float | None = None,
    top_k: int | None = None,
) -> list[tuple[int, int]]:
    """
    Read a Voita head summary CSV (layer, head, accuracy) and
    return a list of (layer, head) pairs.
    """
    df = pd.read_csv(csv_path)
    if "accuracy" not in df.columns:
        raise ValueError(f"{csv_path} must contain an 'accuracy' column.")

    df = df.sort_values("accuracy", ascending=False)

    if accuracy_threshold is not None:
        df = df[df["accuracy"] >= accuracy_threshold]

    if top_k is not None:
        df = df.head(top_k)

    if df.empty:
        raise ValueError(
            f"No heads selected from {csv_path} "
            f"(threshold={accuracy_threshold}, top_k={top_k})."
        )

    heads = [(int(row["layer"]), int(row["head"])) for _, row in df.iterrows()]
    print(f"[HEAD-SELECT] From {csv_path}, selected {len(heads)} heads:")
    print(df[["layer", "head", "accuracy"]])
    return heads


# ---------- entropy core ----------

def compute_entropy_for_sentence(
    sentence: str,
    selected_heads: list[tuple[int, int]],
    tokenizer,
    model,
    max_len: int | None = None,
) -> pd.DataFrame:
    """
    Compute attention entropy for a single sentence:

    - Aggregate attention across selected heads.
    - For token i, compute entropy over attention to previous tokens j < i.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    if max_len is not None and seq_len > max_len:
        # Skip sentences longer than context window
        return pd.DataFrame(columns=["token_idx", "token", "entropy"])

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: list of length L (layers), each (1, heads, seq, seq)
    attn_tensor = torch.stack(outputs.attentions).squeeze(1)  # (L, H, seq, seq)

    # aggregate attention across selected heads into (seq, seq)
    agg_attn = torch.zeros(attn_tensor.shape[-2], attn_tensor.shape[-1], device=DEVICE)
    for layer, head in selected_heads:
        agg_attn += attn_tensor[layer, head]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    rows = []
    for i in range(seq_len):
        # attention from token i to previous tokens
        scores = agg_attn[i, :i]

        if i == 0 or torch.all(scores == 0):
            entropy = float("nan")
        else:
            probs = scores / scores.sum()
            mask = probs > 0
            p = probs[mask]
            entropy = float(-(p * p.log()).sum().item())  # nats

        rows.append(
            {
                "token_idx": i,
                "token": tokens[i],
                "entropy": entropy,
            }
        )

    return pd.DataFrame(rows)


# ---------- model-specific chunk runners ----------

def run_gpt2_entropy_on_chunk(
    df_chunk: pd.DataFrame,
    selected_heads: list[tuple[int, int]],
    sentence_column: str,
    chunk_id: int,
    log_every: int = 100,
) -> pd.DataFrame:
    print("[GPT2-ENTROPY] Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(GPT2_MODEL_NAME, output_attentions=True)
    model.to(DEVICE)
    model.eval()

    max_len = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    print(f"[GPT2-ENTROPY] max_len={max_len}")

    all_rows = []
    total = len(df_chunk)

    for i, row in enumerate(df_chunk.itertuples(), start=1):
        sentence = getattr(row, sentence_column)
        sentence_id = row.Index  # index within chunk

        ent_df = compute_entropy_for_sentence(
            sentence=sentence,
            selected_heads=selected_heads,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
        )

        if ent_df.empty:
            continue

        ent_df["sentence_id"] = sentence_id
        ent_df["sentence"] = sentence
        ent_df["model"] = "gpt2"

        all_rows.append(ent_df)

        if i % log_every == 0:
            print(
                f"[GPT2-ENTROPY] Chunk {chunk_id}: processed {i}/{total} sentences...",
                flush=True,
            )

    if not all_rows:
        return pd.DataFrame(
            columns=["sentence_id", "token_idx", "token", "entropy", "sentence", "model"]
        )

    return pd.concat(all_rows, ignore_index=True)


def run_bert_entropy_on_chunk(
    df_chunk: pd.DataFrame,
    selected_heads: list[tuple[int, int]],
    sentence_column: str,
    chunk_id: int,
    log_every: int = 100,
) -> pd.DataFrame:
    print("[BERT-ENTROPY] Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME, output_attentions=True)
    model.to(DEVICE)
    model.eval()

    max_len = model.config.max_position_embeddings
    print(f"[BERT-ENTROPY] max_len={max_len}")

    all_rows = []
    total = len(df_chunk)

    for i, row in enumerate(df_chunk.itertuples(), start=1):
        sentence = getattr(row, sentence_column)
        sentence_id = row.Index  # index within chunk

        ent_df = compute_entropy_for_sentence(
            sentence=sentence,
            selected_heads=selected_heads,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
        )

        if ent_df.empty:
            continue

        ent_df["sentence_id"] = sentence_id
        ent_df["sentence"] = sentence
        ent_df["model"] = "bert"

        all_rows.append(ent_df)

        if i % log_every == 0:
            print(
                f"[BERT-ENTROPY] Chunk {chunk_id}: processed {i}/{total} sentences...",
                flush=True,
            )

    if not all_rows:
        return pd.DataFrame(
            columns=["sentence_id", "token_idx", "token", "entropy", "sentence", "model"]
        )

    return pd.concat(all_rows, ignore_index=True)


# ---------- chunk driver ----------

def process_chunk(
    df: pd.DataFrame,
    chunk_start: int,
    chunk_end: int,
    chunk_id: int,
    sentence_column: str,
    do_gpt2: bool,
    do_bert: bool,
    gpt2_heads: list[tuple[int, int]] | None,
    bert_heads: list[tuple[int, int]] | None,
    log_every: int,
    output_dir: str,
):
    df_chunk = df.iloc[chunk_start:chunk_end].reset_index(drop=True)
    print(
        f"\n=== [ENTROPY] Processing chunk {chunk_id} "
        f"(rows {chunk_start}–{chunk_end}, {len(df_chunk)} rows) ==="
    )

    os.makedirs(output_dir, exist_ok=True)

    if do_gpt2 and gpt2_heads is not None:
        gpt2_ent = run_gpt2_entropy_on_chunk(
            df_chunk=df_chunk,
            selected_heads=gpt2_heads,
            sentence_column=sentence_column,
            chunk_id=chunk_id,
            log_every=log_every,
        )
        gpt2_path = os.path.join(output_dir, f"gpt2_entropy_chunk_{chunk_id:04d}.csv")
        gpt2_ent.to_csv(gpt2_path, index=False)
        print(
            f"[GPT2-ENTROPY] Chunk {chunk_id}: saved {len(gpt2_ent)} rows to {gpt2_path}"
        )

    if do_bert and bert_heads is not None:
        bert_ent = run_bert_entropy_on_chunk(
            df_chunk=df_chunk,
            selected_heads=bert_heads,
            sentence_column=sentence_column,
            chunk_id=chunk_id,
            log_every=log_every,
        )
        bert_path = os.path.join(output_dir, f"bert_entropy_chunk_{chunk_id:04d}.csv")
        bert_ent.to_csv(bert_path, index=False)
        print(
            f"[BERT-ENTROPY] Chunk {chunk_id}: saved {len(bert_ent)} rows to {bert_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute GPT-2 / BERT attention entropy over selected heads, in chunks."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=N_SUBJ_ROOT_CSV,
        help=f"Input CSV with sentences (default: {N_SUBJ_ROOT_CSV}).",
    )
    parser.add_argument(
        "--sentence-column",
        type=str,
        default="sentence",
        help="Name of the sentence column in input CSV (default: 'sentence').",
    )
    parser.add_argument(
        "--gpt2-heads-csv",
        type=str,
        default=None,
        help="Voita head summary CSV for GPT-2 "
             "(e.g. results/gpt2_voita_head_accuracy_full.csv).",
    )
    parser.add_argument(
        "--bert-heads-csv",
        type=str,
        default=None,
        help="Voita head summary CSV for BERT "
             "(e.g. results/bert_voita_head_accuracy_full.csv).",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=None,
        help="Minimum accuracy for head selection (applied to both models, if provided).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top heads to keep (after threshold; applied to both models, if provided).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows per chunk (default: 5000).",
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
        default=100,
        help="Log progress every N sentences per chunk (default: 100).",
    )
    parser.add_argument(
        "--do-gpt2",
        action="store_true",
        help="Compute entropy for GPT-2.",
    )
    parser.add_argument(
        "--do-bert",
        action="store_true",
        help="Compute entropy for BERT.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save entropy chunk CSVs (default: results).",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional limit on number of sentences loaded from input CSV.",
    )

    args = parser.parse_args()

    if not args.do_gpt2 and not args.do_bert:
        print("[ENTROPY] No model selected. Use --do-gpt2 and/or --do-bert.")
        return

    print(f"[ENTROPY] Loading input sentences from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if args.sentence_column not in df.columns:
        raise ValueError(
            f"{args.input_csv} must contain column '{args.sentence_column}'"
        )

    if args.max_sentences is not None:
        df = df.iloc[:args.max_sentences].reset_index(drop=True)
        print(f"[ENTROPY] Restricting to first {len(df)} sentences.")

    total_rows = len(df)
    print(f"[ENTROPY] Total rows in dataset: {total_rows}")

    # Heads selection
    gpt2_heads = None
    if args.do_gpt2:
        if not args.gpt2_heads_csv:
            raise ValueError("You must provide --gpt2-heads-csv when using --do-gpt2.")
        gpt2_heads = select_heads_from_csv(
            args.gpt2_heads_csv,
            accuracy_threshold=args.accuracy_threshold,
            top_k=args.top_k,
        )

    bert_heads = None
    if args.do_bert:
        if not args.bert_heads_csv:
            raise ValueError("You must provide --bert-heads-csv when using --do-bert.")
        bert_heads = select_heads_from_csv(
            args.bert_heads_csv,
            accuracy_threshold=args.accuracy_threshold,
            top_k=args.top_k,
        )

    # Chunk loop
    start_row = args.skip_chunks * args.chunk_size
    if start_row >= total_rows:
        print(
            f"[ENTROPY] skip_chunks={args.skip_chunks} (start_row={start_row}) "
            f"is beyond dataset size."
        )
        return

    for k in range(args.num_chunks):
        chunk_id = args.skip_chunks + k
        chunk_start = start_row + k * args.chunk_size
        if chunk_start >= total_rows:
            print(f"[ENTROPY] Reached end of dataset at chunk {chunk_id}. Stopping.")
            break

        chunk_end = min(chunk_start + args.chunk_size, total_rows)
        process_chunk(
            df=df,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_id=chunk_id,
            sentence_column=args.sentence_column,
            do_gpt2=args.do_gpt2,
            do_bert=args.do_bert,
            gpt2_heads=gpt2_heads,
            bert_heads=bert_heads,
            log_every=args.log_every,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
