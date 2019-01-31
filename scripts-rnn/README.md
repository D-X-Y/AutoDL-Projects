# Search RNN cell
```
bash scripts-nas-rnn/search-baseline.sh 3
bash scripts-nas-rnn/search-accelerate.sh 0 200 10 1
```

# Train the Searched Model
```
bash scripts-nas-rnn/train-PTB.sh 3 DARTS_V1
bash scripts-nas-rnn/train-WT2.sh 3 DARTS_V1
bash scripts-nas-rnn/train-PTB.sh 3 DARTS_V2
```
