# Dynamic-Clip-Attention
Implementation of A Compare-Aggregate Model with Dynamic-Clip Attention for Answer Selection on TrecQA and WikiQA using Keras

## Prerequisites
- Python 2.x
- Theano 0.8
- Keras  1.0

## Data
- [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- [TrecQA: Answer Selction Task](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Getting Started
First, you need to install dependencies, then clone this repo:
```
git clone https://github.com/wjbianjason/Dynamic-Clip-Attention
```

I have uploaded my prepocess result of dataset, you can repeat the procedure as follow:
<br/>
### WikiQA Preprocess
**Note**: dowload \"WikiQACorpus.zip\" to the path "./data/raw_data/WikiQA/".
>WikiQACorpus.zip download link: https://www.microsoft.com/en-us/download/details.aspx?id=52419
```
sh preprocess.sh wikiqa
```
### TrecQA Preprocess
**Note**: If you don't have svn command, you can copy the directory [TrecQA_of_CIKM2016_Rao](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA) to our path "./data/raw_data/"
```
sh preprocess.sh trecqa
```

Because I have uploaded my preprocess data, you can skip above operations.

## Running

```
usage: main.py [-h] [-t TASK] [-m MODEL] [-d HIDDEN_DIM] [-e EPOCH] [-l LR]
               [-k_a K_VALUE_QUES] [-k_q K_VALUE_ANS] [-b BATCH_SIZE]
               [-r RANDOM_SIZE]
```

### WikiQA
Basic approach: **listwise**
```
python main.py -t wikiqa -m listwise -d 300 -e 10 -l 0.001 -b 5 -r 15
```
**Note**: k_max and k_threshold need basic approach trained model to init weights.
So please running basic approach first.
<br/>
Second approach: **k_max**
```
python main.py -t wikiqa -m k_max -d 300 -e 5 -l 0.001 -b 3 -r 15 -k_q 5 -k_a 10
```
Third approach: **k_threshold**
```
python main.py -t wikiqa -m k_threshold -d 300 -e 5 -l 0.001 -b 3 -r 15 -k_q 0.1 -k_a 0.05
```


### TrecQA
Basic approach: **listwise**
```
python main.py -t trecqa -m listwise -d 300 -e 10 -l 0.001 -b 3 -r 50
```
**Note**: k_max and k_threshold need basic approach trained model to init weights.
So please running basic approach first.
<br/>
Second approach: **k_max**
```
python main.py -t trecqa -m k_max -d 300 -e 5 -l 0.001 -b 3 -r 50 -k_q 5 -k_a 10
```
Third approach: **k_threshold**
```
python main.py -t trecqa -m k_threshold -d 300 -e 5 -l 0.001 -b 3 -r 50 -k_q 0.1 -k_a 0.05
```

## Results
In all experiments, we selected training models that obtain the best MAP scores on the development set for testing.
<br/>
You should be able to reproduce some scores close to the numbers in the experiment table of our paper.
<br/>
If you want to reproduce the same score, you need to use the following command:
```
THEANO_FLAGS="dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic" python
```
which makes the cuDNN's backward pass is deterministic. This is a reproduce problem for Theano, not our trick. 


## Copyright
Copyright Author @wjbianjason. All Rights Reserved.