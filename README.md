# Cross-linguistic Effects of Syncretism on Agreement Attraction

This repository contains code and data for quantifying how case syncretism affects agreement attraction across languages using large language models (LLMs).

## Research Context

**Agreement attraction** occurs when speakers mistakenly accept or produce sentences like *"The key to the cabinets are rusty"* due to interference from a plural "attractor" noun (*cabinets*), even though the grammatical subject (*key*) is singular.

**Syncretism** refers to ambiguous case marking—when a single morphological form maps to multiple grammatical cases (e.g., English *"you"* is both nominative and accusative).

### Cross-linguistic Puzzle

Languages differ in how syncretism affects attraction:

| Language | Syncretism Effect | Example |
|----------|-------------------|---------|
| **English** | Distinctive case marking reduces attraction | *"elves' garden"* (genitive) vs *"cabinets"* (ambiguous) |
| **Russian** | Syncretic forms increase attraction | GEN.SG = NOM.PL forms cause more errors |
| **Turkish** | No sensitivity to syncretism | Distinctive marking doesn't help |

### Approach

Following [Ryu & Lewis (2021)](https://aclanthology.org/2021.acl-long.12/), we use:
1. **Attention values** from BERT/GPT-2 as a proxy for memory retrieval likelihood
2. **Surprisal values** as a proxy for reading times

We probe which attention heads best predict nsubj→root dependencies using large corpora, then extract attention/surprisal on experimental stimuli from psycholinguistic studies.

## Models

| Language | BERT | GPT-2 |
|----------|------|-------|
| English | `bert-base-uncased` | `gpt2` |
| German | `bert-base-german-cased` | `dbmdz/german-gpt2` |
| Russian | `deepvk/bert-base-uncased` | `ai-forever/rugpt3small_based_on_gpt2` |
| Turkish | `dbmdz/bert-base-turkish-128k-cased` | `redrussianarmy/gpt2-turkish-cased` |

## Repository Structure

```
├── llm_code/
│   ├── english_attention/    # English analysis pipeline
│   ├── german_attention/     # German analysis pipeline
│   ├── russian_attention/    # Russian analysis pipeline
│   └── turkish_attention/    # Turkish analysis pipeline
│       ├── config.py                 # Model names, paths, URLs
│       ├── corpus2conllu.py          # Download & parse Leipzig corpus
│       ├── build_dataset.py          # Extract nsubj-root pairs
│       ├── voita_run_attention.py    # Voita-style head probing
│       ├── analyze_attention.py      # Attention on experimental stimuli
│       ├── analyze_surprisal.py      # Surprisal on experimental stimuli
│       └── utils.py                  # Shared utilities
├── data/                     # Corpora and extracted datasets
├── stimuli/                  # Experimental stimuli from cited studies
├── stats_code/               # R scripts for statistical analysis
├── figures/                  # Generated figures
└── text/                     # Paper (main.tex)
```

## Large CSV Handling

The `data/nsubj_root_sentences_*.csv` files are large. Use the helper scripts to split
them into 50MB chunks and merge them back later.

Split into 50MB parts (default):
```bash
bash scripts/split_nsubj.sh 50
```

Split into 50MB parts in a separate directory:
```bash
bash scripts/split_nsubj.sh 50 data/parts
```

Merge parts back into full CSVs:
```bash
bash scripts/merge_nsubj.sh data/parts data
```

## Pipeline

### Step 1: Data Preparation (Complete)

Download Leipzig corpora and parse with UDPipe to get CoNLL-U format:
```bash
cd llm_code/[lang]_attention
python corpus2conllu.py
```

Extract nsubj-root dependency pairs:
```bash
python build_dataset.py
```

### Step 2: Voita Head Probing (Complete)

Identify which attention heads best predict nsubj→root dependencies:
```bash
python voita_run_attention.py
```

This processes ~1M+ sentences per language and outputs per-head accuracy scores.

**Status:** Complete for all 4 languages (BERT + GPT-2) as of 2026-01-19.

### Step 3: Attention & Surprisal on Stimuli (Complete)

Extract metrics on experimental sentences:
```bash
python analyze_attention.py
python analyze_surprisal.py
```

**Status:** Complete for all 4 languages as of 2026-01-19.

### Step 4: Statistical Analysis

Run R scripts for Bayesian mixed-effects models:
```bash
cd stats_code
Rscript data_prep.R
Rscript models.R
Rscript plotting.R
```

## Current Results

From the paper (CogSci submission):

| Language | Attention | Surprisal | Syncretism Effect |
|----------|-----------|-----------|-------------------|
| English | Predicted | Predicted | Replicated |
| Russian | Predicted | Partial | Not replicated |
| Turkish | Not conclusive | Predicted | Insensitivity confirmed |

---

## Project Log

### 2026-01-19: Full Pipeline Execution via SLURM

Ran all pipelines on GPU cluster (V100) via SLURM batch jobs:

**Completed:**
- Voita head probing for BERT and GPT-2 across all 4 languages (~5.5M sentences total)
- Attention analysis on experimental stimuli
- Surprisal analysis on experimental stimuli

**Key Finding:** After fixing BERT directionality to use `nsubj → root` (dependent → head), BERT head accuracy dropped to near-zero (<0.01%), while GPT-2 heads achieve 30-52% accuracy. This suggests BERT's bidirectional attention is too diffuse for single-head dependency prediction.

| Language | Best BERT Head | Best GPT-2 Head |
|----------|----------------|-----------------|
| English | 0.01% | 49.6% |
| German | 0.01% | 38.0% |
| Russian | 0.01% | 52.1% |
| Turkish | 0.01% | 29.7% |

**Results Generated:**
- `llm_code/[lang]_attention/results/bert_voita_head_accuracy_full.csv`
- `llm_code/[lang]_attention/results/gpt2_voita_head_accuracy_full.csv`
- `llm_code/[lang]_attention/results/[lang]_attention_results.csv`
- `llm_code/[lang]_attention/results/[lang]_surprisal_results.csv`

**Remaining:** R statistical analysis, paper updates.

---

### Limitations & Next Steps

1. **BERT head accuracy near zero**: After directionality fix, BERT heads don't predict nsubj→root well. May need multi-head aggregation or alternative probing methods.
2. **Single-head assumption**: Agreement likely involves multiple heads encoding case, number, and position.
3. **Counterfactual intervention**: Future work could use methods from [Hao & Linzen (2023)](https://aclanthology.org/2023.acl-long.38/) to manipulate representations.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `transformers`, `torch`, `pandas`, `conllu`


## References

- Wagers et al. (2009) - Cue-based retrieval model
- Nicol et al. (2016) - English possessive syncretism
- Slioussar (2018) - Russian case syncretism
- Lago et al. (2019) & Turk & Logacev (2024) - Turkish attraction
- Ryu & Lewis (2021) - LLM attention/surprisal for attraction
