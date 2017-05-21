# Dynamic-Clip-Attention
Implementation of [A COMPARE-AGGREGATE MODEL WITH DYNAMIC-CLIP ATTENTION FOR ANSWER SELECTION][paper] on TrecQA and WikiQA using Keras

## Prerequisites
- Python 2.x
- Theano 0.8
- Keras  1.0

## Data
- [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- [TrecQA clean: Answer Selction Task uploaded by CIKM2016 Rao](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Getting Started
First, you need to install dependencies, then clone this repo:
```
git clone https://github.com/wjbianjason/Dynamic-Clip-Attention
```

I have uploaded my prepocess result of dataset, you can repeat the procedure as follow:
WikiQA Preprocess
Note:dowload \"WikiQACorpus.zip\" to the path ./data/raw_data/WikiQA/ through address: https://www.microsoft.com/en-us/download/details.aspx?id=52419
```
sh preprocess.sh wikiqa
```
TrecQA Preprocess
Note:If you don't have svn command, you can copy the directory [https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA) to our path /data/raw_data/
```
sh preprocess.sh trecqa
```

because I have upload my preprocess data, so you can skip above operation.

## Running

```
usage: main.py [-h] [-t TASK] [-m MODEL] [-d HIDDEN_DIM] [-e EPOCH] [-l LR]
               [-k_a K_VALUE_QUES] [-k_q K_VALUE_ANS] [-b BATCH_SIZE]
               [-r RANDOM_SIZE]
```

WikiQA
basic approach:listwise
```
python main.py -t wikiqa -m listwise -d 300 -e 15 -l 0.001 -b 3 -r 15
```
Notice: k-max and k-threshold need basic approach trained model to init weights.
So please running basic approach first.
sencond approach:k-max
```
python main.py wikiqa k-max
```
third approach:k-threshold
```
python main.py wikiqa k-threshold
```


TrecQA
basic approach:listwise
```
python main.py trecqa basic_listwise
```
Notice: k-max and k-threshold need basic approach trained model to init weights.
So please running basic approach first.
sencond approach:k-max
```
python main.py trecqa k-max
```
third approach:k-threshold
```
python main.py trecqa k-threshold
```

## Results
-------
You should be able to reproduce some scores close to the numbers in the experiment table of our paper:
If you want to reproduce the same score, you need to use the following command:
```
THEANO_FLAGS="dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic" python
```
which makes the cuDNN's backward pass is deterministic. This is a reproduce problem for Theano, not my trick. 


## Copyright
Copyright 2017 PRIS of BUPT. All Rights Reserved.