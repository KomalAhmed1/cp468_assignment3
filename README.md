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


**Source:** SQuAD 1.1, via the Du et al. (2017) "Learning to Ask" split (github.com/xinyadu/nqg), which partitions SQuAD's articles (not just questions) into train/dev/test. Because the split is at the article level, no paragraph or question from a test article ever leaks into training — this satisfies "reserve held-out validation and test sets before any model development" and avoids the leakage a naive question-level split risks.

**License:** SQuAD is released under CC BY-SA 4.0. Cite: Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text, EMNLP 2016; and Du, Shao & Cardie (2017), Learning to Ask: Neural Question Generation for Reading Comprehension, ACL 2017 (for the split).

**Counts after flattening/filtering (src/preprocess.py): **75,711 train / 10,570 val / 11,877 test question-generation examples.

**Input representation:** Passage tokens with the answer span wrapped in < ans > ... < ans > markers, truncated to a ±25-token window around the answer

**Parameters: **   6,484,560
**Hardware:**	1 CPU core, 1 GPU
**Train / val example:s**	5,000 / 500 (subsampled from the full 75,711/10,570 with --max_train/--max_val)
**Epochs:**	8
**Total training time:**	1,199 s (~20 min)
**Best val loss:**	5.21 (epoch 2)
