import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from pickle import (
    load as pload,
    dump as pdump
)
import sys    
tf.set_random_seed(7171)
CELLNAMES = ["BasicRNN", "BasicLSTM", "GRU"]

class RNN:
    def __init__(
        self, input_data, tags, cell, w2v_model_path, 
        hidden_size=2, output_dimension=1,
        activation_fn=tf.tanh, learning_rate=0.001):

        # Embedding Vector
        self.base_w2v = Word2Vec.load(w2v_model_path)
        #self.input_w2v = Word2Vec(input_data, size=self.base_w2v.vector_size, window=self.base_w2v.window)
        self.hidden_size = hidden_size        
        self.learning_rate = learning_rate
        self.input_data = input_data
        
        self.y = np.array([[int(t)] for t in tags])
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

        self.x_train, self.x_validation, self.x_test, self.y_train, self.y_validation, self.y_test, self.sl_train, self.sl_validation, self.sl_test = self.train_valid_test_split()
        print("Placerholder init...")         
        self.X = tf.placeholder(
            tf.float32, 
            [None, max(self.sequence_length), self.base_w2v.vector_size],
            name='X'
        )
        self.Y = tf.placeholder(
            tf.float32,
            [None, 1],
            name='Y'
        )
        self.sl = tf.placeholder(
            tf.float32,
            [None],
            name="Sequence_Length"
        )
        print("RNN settings...")
        # RNN Settings
        self.cell = cell(num_units=self.hidden_size)
        #self.initial_state = self.cell.zero_state(batch_size, dtype=tf.float32)
        self._output, self._state = tf.nn.dynamic_rnn(
            self.cell, 
            self.X,
            #initial_state = self.initial_state,
            dtype = tf.float32,
            sequence_length = self.sl,
        )
        # FCC Layer
        self.Y_pred = tf.contrib.layers.fully_connected(
            self._state, 
            output_dimension, 
            activation_fn=None 
        )
        self.pred_squeeze = tf.squeeze(self.Y_pred)
        self.prediction = tf.cast(self.pred_squeeze > 0.5, dtype=tf.float32)
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = self.Y,
            logits = self.Y_pred,
            name = "sigmoid_cross_entropy"
        )
        self._loss = tf.reduce_mean(self.cross_entropy, name="loss")
        self._train = tf.train.AdamOptimizer(learning_rate).minimize(self._loss, name="train")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.squeeze(self.Y)), dtype=tf.float32))
         
   
    def train_valid_test_split(self, ratio=[0.6, 0.2, 0.2], random_state=42):
        test_size = ratio[2]
        validation_size = ratio[1] / float(ratio[0]+ratio[1])
        # test set split
        x_train_plus_validation, x_test, y_train_plus_validation, y_test = train_test_split(self.vectorized, self.y, test_size=test_size, random_state=random_state)
        sl_train_plus_validation, sl_test = train_test_split(self.sequence_length, test_size=test_size, random_state=random_state)
        # validation set split
        x_train, x_validation, y_train, y_validation = train_test_split(x_train_plus_validation, y_train_plus_validation, test_size=validation_size, random_state=random_state)
        sl_train, sl_validation = train_test_split(sl_train_plus_validation, test_size = validation_size, random_state=random_state)
        return x_train, x_validation, x_test, y_train, y_validation, y_test, sl_train, sl_validation, sl_test

    def get_total_num_params(self, trainable_variables):
        total_params = 0
        for variable in trainable_variables:
            shape = variable.get_shape()
            variable_params = 1
            for dim in shape:
               variable_params *= dim.value
            total_params += variable_params
        return total_params
    
    def run(self, epoch = 10, batch_size = 100, cidx=2, logfile_prefix="", session_freeze_prefix=""):
        description = 'hsize%d_alpha%.5f_%s_epoch%d_batchsize%d_w2v_window%d_size%d_mincount%d' % (self.hidden_size, self.learning_rate, CELLNAMES[cidx], epoch, batch_size, self.base_w2v.window, self.base_w2v.vector_size, self.base_w2v.min_count)     
        logfile = '/'.join((logfile_prefix, description)) + '.txt'
        session_dump = '/'.join((session_freeze_prefix, description)) 

        f = open(logfile, 'w+')
        total_batch = len(self.vectorized)
        train_total_batch = len(self.x_train)
         
        saver = tf.train.Saver()
        print("session starts..!")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(self.get_total_num_params(tf.trainable_variables()))
            prev_epoch_accuracy = -1
            for ep in range(epoch):
                print('epoch : %d' % ep)
                # training data set
                cnt = 0; loss_sum = 0; acc_sum = 0
                for i in range(train_total_batch // batch_size):
                    start, end = i * batch_size , (i+1) * batch_size
                    X_batch = self.x_train[start:end]
                    y_batch = self.y_train[start:end]
                    sl_batch = self.sl_train[start:end]
                    output, pred, train_loss, train, train_accuracy = sess.run([self._output, self.prediction, self._loss, self._train, self.accuracy], feed_dict={self.X:X_batch, self.Y:y_batch, self.sl:sl_batch})
                    
                    loss_sum += train_loss
                    acc_sum += train_accuracy
                    cnt += 1
                avg_loss = loss_sum / float(cnt)
                avg_accuracy = acc_sum / float(cnt)
                
                # test data set
                test_loss, test_accuracy = sess.run([self._loss, self.accuracy], feed_dict={self.X:self.x_test, self.Y:self.y_test, self.sl:self.sl_test})
                # validation data set
                validation_loss, validation_accuracy = sess.run([self._loss, self.accuracy], feed_dict={self.X:self.x_validation, self.Y:self.y_validation, self.sl:self.sl_validation})
                print(ep)
                print("train      :", avg_loss, avg_accuracy)
                print("test       :", test_loss, test_accuracy)
                print("validation :", validation_loss, validation_accuracy)
                f.write(' '.join((str(ep),str(avg_loss),str(avg_accuracy),str(test_loss),str(test_accuracy), str(validation_accuracy),"\n")))

            saver.save(sess, session_dump)
        f.close()

def load_data():
    input_setences, tags = None, None
    with open("./dumps/naver_movies_sentences.bin", 'rb') as fp:
        input_setences = pload(fp)
    with open("./dumps/tags.bin", 'rb') as fp:
        tags = pload(fp)
    
    if not input_setences or not tags:
        raise TypeError

    return input_setences, tags

def get_argv():
    hidden_size, learning_rate, epoch, batch_size, cell_index = sys.argv[1:]
    return int(hidden_size), float(learning_rate), int(epoch), int(batch_size), int(cell_index)
 
def run():
    basicCell, LSTMCell, GRUCell = tf.nn.rnn_cell.BasicRNNCell, tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.GRUCell
    cells = [basicCell, LSTMCell, GRUCell]
    input_sentences, tags = load_data()
    hsize, alpha, epoch, batch_size, cidx = get_argv()
    model = RNN(
       input_sentences,
        tags, 
        cells[cidx], 
        "./models/w2v_window4_size50_mincount40.model", 
        output_dimension=1,
        hidden_size=hsize,
        learning_rate=alpha
    )
    model.run(
        epoch=epoch, 
        batch_size=batch_size, 
        cidx=cidx, 
        logfile_prefix='./logs', 
        session_freeze_prefix = "./session_freeze")

if __name__ == '__main__':
    run()
    print("FINISHED!")
