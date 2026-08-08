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
