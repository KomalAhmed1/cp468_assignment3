# cp468_assignment3

**STEP 1: import the raw SQuAD data use:**
mkdir -p data/raw
cd data/raw
curl -sL -o train.json https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/train.json
curl -sL -o dev.json   https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/dev.json
curl -sL -o test.json  https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/test.json
cd ../..

**STEP 2: install dependencies**
pip install torch numpy sacrebleu rouge-score nltk

**STEP 3: Preprocess**
cd src
python preprocess.py --raw_dir ../data/raw --out_dir ../data/processed

**STEP 4: Train the LSTM**
python train.py --max_train 5000 --max_val 500 --epochs 8 \
    --batch_size 128 --emb_dim 112 --hidden_dim 160 --lr 0.001 --tf_ratio 0.6 --seed 13

**STEP 5: Decode LSTM predictions on the test set**
python generate.py --split test --n_examples 500 --decode greedy --out ../results/lstm_predictions.jsonl

**STEP 6: Score the LSTM**
python evaluate.py --pred ../results/lstm_predictions.jsonl --name lstm 

**STEP 7: LLM baseline on the identical test set (two prompt variants)**
Same `--split/--n_examples/--seed` as STEP 5, so `load_jsonl` samples the same 500 rows
(verified: the qid sets of both LLM runs are equal to the LSTM's).

Reported runs use a locally hosted open-weights model, so the baseline is reproducible
with no API key and no cost:
brew install ollama && ollama serve &
ollama pull llama3.1:8b

python llm_baseline.py --prompt zeroshot --n_examples 500 \
    --base_url http://localhost:11434/v1 --api_key_env "" --model llama3.1:8b \
    --price_in 0 --price_out 0 --out ../results/llm_zeroshot_predictions.jsonl
python llm_baseline.py --prompt fewshot --k_shots 4 --n_examples 500 \
    --base_url http://localhost:11434/v1 --api_key_env "" --model llama3.1:8b \
    --price_in 0 --price_out 0 --out ../results/llm_fewshot_predictions.jsonl

Any OpenAI-compatible API works instead, e.g. Gemini:
export LLM_API_KEY=...
python llm_baseline.py --prompt zeroshot --n_examples 500 --sleep 2 \
    --base_url https://generativelanguage.googleapis.com/v1beta/openai \
    --model gemini-3.1-flash-lite --reasoning_effort none \
    --out ../results/llm_zeroshot_predictions.jsonl
(`--reasoning_effort none` is required for Gemini thinking models, which otherwise
spend the whole token budget on reasoning and return empty content. The free tier caps
at 500 requests/day, so a 2x500 run needs `--resume` across two days or a paid key.)

Each run writes `results/llm_<variant>_runmeta.json` with the exact prompts, token
counts, estimated USD cost, wall-clock/compute hours and median latency.

**STEP 8: Score the LLM (same scorer as the LSTM)**
python evaluate.py --pred ../results/llm_zeroshot_predictions.jsonl --name llm_zeroshot
python evaluate.py --pred ../results/llm_fewshot_predictions.jsonl  --name llm_fewshot

Few-shot demonstrations are drawn from `train.jsonl` only — never val/test — so the
held-out sets stay clean.

**LLM baseline as run (Section 4.2)**

| | zero-shot | few-shot (k=4) |
|---|---|---|
| Model | llama3.1:8b (Ollama 0.32.6, local) | llama3.1:8b (Ollama 0.32.6, local) |
| Test examples | 500 (identical qids to LSTM) | 500 (identical qids to LSTM) |
| Prompt / completion tokens | 123,505 / 7,953 | 809,505 / 7,571 |
| Empty responses | 0 | 0 |
| Wall clock | 736.0 s (0.204 h) | 796.2 s (0.221 h) |
| Median latency | 1.41 s | 1.52 s |
| API cost (USD) | $0.00 (local) | $0.00 (local) |

Cost is therefore ~0.43 CPU/GPU-hours total on the hardware listed above, and $0 in API
spend. Decoding is greedy-equivalent (`--temperature 0`).


**Source:** SQuAD 1.1, via the Du et al. (2017) "Learning to Ask" split (github.com/xinyadu/nqg), which partitions SQuAD's articles (not just questions) into train/dev/test. Because the split is at the article level, no paragraph or question from a test article ever leaks into training — this satisfies "reserve held-out validation and test sets before any model development" and avoids the leakage a naive question-level split risks.

**License:** SQuAD is released under CC BY-SA 4.0. Cite: Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text, EMNLP 2016; and Du, Shao & Cardie (2017), Learning to Ask: Neural Question Generation for Reading Comprehension, ACL 2017 (for the split).

**Counts after flattening/filtering (src/preprocess.py):** 75,711 train / 10,570 val / 11,877 test question-generation examples.

**Input representation:** Passage tokens with the answer span wrapped in < ans > ... < ans > markers, truncated to a ±25-token window around the answer

**Parameters:**   6,484,560

**Hardware:**	1 CPU core, 1 GPU

**Train / val example:s**	5,000 / 500 (subsampled from the full 75,711/10,570 with --max_train/--max_val)

**Epochs:**	8

**Total training time:**	1,199 s (~20 min)

**Best val loss:**	5.21 (epoch 2)
