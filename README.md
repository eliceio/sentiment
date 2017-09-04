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
      * Load Word2Vec model for matched hyperparameters
      ```python
      ...
      self.base_w2v = Word2Vec.load(w2v_model_path)
      ...
      ```
      * Add Paddings for each seqeuences
      ```python
      ...
      self.sequence_length = [len(seq) for seq in input_data]
      self.max_seq_len = max(self.sequence_length)
      print(self.max_seq_len)
      self.sentences = [s for s in input_data] 
      self.vectorized = []
      for sentence in input_data:
          cnt = 0; s = []
          for word in sentence:
              try:
                  s.append(self.base_w2v[word])
              except:
                  s.append(np.zeros(self.base_w2v.vector_size, dtype="float32")) 
              cnt += 1
          for i in range(self.max_seq_len - cnt):
              s.append(np.zeros(self.base_w2v.vector_size, dtype="float32"))
          self.vectorized.append(s)
      ...
      ```
      * GRU Layer
      ```python
      ...
      self.cell = cell(num_units=self.hidden_size) # cell is initialized with GRU when model instanciated
      self._output, self._state = tf.nn.dynamic_rnn(
          self.cell, 
          self.X,
          #initial_state = self.initial_state,
          dtype = tf.float32,
          sequence_length = self.sl,
      )
      ```
      * FC layer
      ```python
      ...
      self.Y_pred = tf.contrib.layers.fully_connected(
          self._state, 
          output_dimension, 
          activation_fn=None 
      )
      ...
      ```
      * Sigmoid Cross Entropy
      ```python
      ...
      self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
          labels = self.Y,
          logits = self.Y_pred,
          name = "sigmoid_cross_entropy"
      )
      ...
      ```
      * Prediction
      ```python
      ...
      self.pred_squeeze = tf.squeeze(self.Y_pred)
      self.prediction = tf.cast(self.pred_squeeze > 0.5, dtype=tf.float32)
      ...
      ```

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

