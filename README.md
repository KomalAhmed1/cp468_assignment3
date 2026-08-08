# cp468_assignment3
**to import the raw SQuAD data use:** 
mkdir -p data/raw
cd data/raw
curl -sL -o train.json https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/train.json
curl -sL -o dev.json   https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/dev.json
curl -sL -o test.json  https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/test.json
cd ../..
