# syncretism_crossling_llm
Exploring the relationship between speakers' and language models' attention to case marking.


## How to run

- `cd llm_code/turkish_attention`
- run `python corpus2conllu.py` to download 1M sentence corpus from leipzig and prase it with Universal Dependencies 
- run `python build_dataset.py` to find nsubj-root pairs in the Turkish UD treebanks and 1M sentence corpus.
- run `python run_attention.py` to get the attention values for nsubj marked elements.
- run `python summarize_attention.py` to get top 5 attention heads. 
