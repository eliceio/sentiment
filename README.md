# Hangul Sentiment Analysis

### Summary
   Hangul sentiment analysis based on trained model using Wiki dumps and Naver movie reviews with tensorflow ANN modules. 

### Paper Study
   1. (Word Embedding 1) [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
   2. (Word Embedding 2) [GloVe: Global Vectors for Word Representation](http://www.aclweb.org/anthology/D14-1162)
   3. (Sentiment Analysis : RNN) [Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)
   4. (Sentiment Analysis : CNN) [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
   5. (Sentiment Analysis : Recursive Net) [Recursive deep models for semantic compositionality over a sentiment treebank](http://www.anthology.aclweb.org/D/D13/D13-1170.pdf) 
   6. (GRU) [Learning phrase representations using RNN encoder-decoder for statistical machine translation ](https://arxiv.org/abs/1406.1078)

### Data Collecting and Preprocessing
   

### Word Embedding


### CNN


### RNN
   * Used Modules
      * tensorflow
      * numpy 
      * sklearn.cross_validation
      * gensim.models.Word2Vec
   * Flow
      * GRU Layer
      * FC layer
      * Sigmoid Cross Entropy
      * Predition

### DEMO
   * [DEMO Site](http://elice-guest-ds-04.koreasouth.cloudapp.azure.com:8000)

### Directory Structure explained 

```
.
├── data_collection
│   └── naver_review.py
├── demo                            - demo server
|   ├── ...
├── model
│   ├── model_cnn.py
│   ├── model_svm.py
│   └── model_w2v.py
├── README.md
├── treeview.txt
└── word2vec_models
    ├── dumps                       - store location for binary dumps for data
    ├── logs                        - store location for logs
    ├── models                      - store location for sentiment models
    ├── parsing.py                  - Data Preprocessing
    ├── requirements.txt            - Virtualenv requirements.txt 
    ├── session_freeze              - freezed session files for Tensorflow
    ├── simpleRNN.py                - RNN model
    ├── w2v_models                  - store location for word2vec models 
    ├── wat.py                      - word analogy reasoning task script
    ├── wats                        - word analogy reasoning tasks word set
    └── word2vec_model_generator.py - word2vec model generator for each set hyper parameters
```

### Team Members
   * Sungjoon Park
   * Nayoun Seo
   * Jaysok Park
   * Jaeyoon Kim
   * Wonki Kim
   * Kuhyun Jung
   * Taehyun Kim

