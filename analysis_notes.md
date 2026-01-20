# Notes on Methods and Implementation Choices

## BERT surprisal consistency ✅ RESOLVED

~~Right now the BERT surprisal logic is not consistent across languages. English/German/Turkish use masked cloze scoring, while Russian uses unmasked logits.~~

**Status:** All four languages now use PLL (pseudo-log-likelihood) for BERT surprisal:
- Mask all verb subtokens simultaneously
- Compute -log P(original token | masked context) for each subtoken
- Sum across subtokens

This is the standard approach from Salazar et al. (2020) and is consistent across English, German, Russian, and Turkish.

## Head probing method (Voita-style)

Using per-head accuracy with argmax attention to detect nsubj->root aligns with prior work (e.g., Voita et al., Clark et al.). The modest accuracies you observed (e.g., 12-36%) are typical in the literature and are not necessarily a bug.

One possible add-on is to explicitly frame this as "head specialization" rather than "full parsing ability," since Htut et al. show attention heads are weak parsers even when some heads align with dependencies.

## Turkish attention word indices and head selection ✅ RESOLVED

Given your fixed-template stimuli, using fixed word indices for head/attractor/verb is defensible. The risk is not the template itself, but small variations that may not be obvious:
- punctuation attached to words (e.g., commas) can change the whitespace-based index,
- optional adverbs or extra modifiers can shift the position of the verb,
- tokenization plus case marking can lead to subtle alignment errors.

~~The bigger issue is head selection: Turkish uses `turkish_bert_attention_nsubj_root.csv` and picks the most frequent "best head" per sentence, while other languages use Voita top-k accuracy. This means Turkish attention is not directly comparable to English/German/Russian.~~

**Status:** Turkish `analyze_attention.py` now uses the same Voita accuracy-based head selection as other languages:
- Reads from `results/bert_voita_head_accuracy_full.csv` (same as English/German/Russian)
- Uses `get_best_heads()` function from `utils.py`
- Selects top-k heads by accuracy, not by frequency

All four languages now use identical head selection methodology.

## Model choices (quick check)

English/German/Turkish choices look standard. The only one I would double-check is Russian BERT: `deepvk/bert-base-uncased` is less common than `DeepPavlov/rubert-base-cased` or `ai-forever/ruBert-base`. It is probably fine, but worth verifying it is the intended model and that it handles Cyrillic as expected.

## Attention entropy as an additional metric

Ryu and Lewis emphasize attention diffuseness (entropy), not only attention to specific targets. You already compute entropy in the attention scripts, but if it is not used downstream, consider surfacing it explicitly in the analysis or at least reporting it as a secondary metric.

## Single best head vs top-k

Using a single best head is a known limitation. Many papers either analyze all heads or use top-k heads / weighted combinations. If you keep single-head selection, it is worth stating explicitly that agreement information is distributed and that single-head results are a proxy.

## Voita directionality ✅ RESOLVED

~~The current Voita direction logic always sets the source token to be the later token in the sentence and the target to be the earlier token. That helps GPT-2, since it can only attend leftward, but for BERT it is a stronger constraint than necessary and differs from the usual "dependent -> head" definition used in many syntax probes.~~

**Status:** Directionality is now model-appropriate across all four languages:

**BERT (all languages):**
- Uses `dependent → head` direction (`nsubj → root`)
- This is the standard linguistic convention for dependency probing
- Allows BERT to use its full bidirectional attention capability

**GPT-2 (all languages):**
- Uses leftward constraint (`source = later token, target = earlier token`)
- Required because GPT-2 is causal and can only attend to previous tokens

This change means BERT and GPT-2 use methodologically distinct (and appropriate) probing directions.

## Other possible checks (optional)

If you want a broader literature alignment section, you could mention:
- bidirectional probing (testing both directions, as in Clark et al.),
- full-tree extraction from attention (e.g., maximum spanning tree; Htut et al.),
- layer-wise analysis (syntax often peaks in mid layers),
- representation interventions (e.g., Hao and Linzen) as future work.

# To-do

## Completed ✅

- [x] **Align Turkish code with others** - Turkish now uses same Voita accuracy-based head selection
- [x] **Fix BERT directionality** - BERT now uses `nsubj → root` (dependent → head) direction
- [x] **Make everything PLL** - All languages use PLL for BERT surprisal (Russian was fixed)
- [x] **Add GPT2 support everywhere** - All languages now have both BERT and GPT2 in:
  - `voita_run_attention.py` (English was BERT-only, now has GPT2)
  - `analyze_attention.py` (Russian and Turkish were BERT-only)
  - `analyze_surprisal.py` (already had both)

## Remaining

- [ ] Verify Russian BERT model (`deepvk/bert-base-uncased` vs `DeepPavlov/rubert-base-cased`)
- [ ] Decide whether to report attention entropy explicitly
- [ ] Run all analysis scripts to verify they work end-to-end
- [ ] Re-run Voita analysis with new directionality (BERT results will change)