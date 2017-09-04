
### train_features, train_target 리스트 설정 ###

train_file = open('./ratings_train.txt', encoding = 'utf-8')

train_text = []
train_features = []
train_target = []

prac = 0 
for line in train_file:
	if prac == 8:
		break
	prac = prac + 1
	train_text.append(line)

del train_text[0]

for line in train_text:
	train_features.append(line.split('\t')[1])
	train_target.append(int(line.split('\t')[2].split('\n')[0]))
	# print(train_target)
	# target_str = line.split('\t')[2].split('\n')[0]
	# train_target.append(int(target_str))

print("train_features :  ", train_features[0])
print("train_target :  ", train_target[0])

### test_feature, test_target 리스트 설정 ###

test_file = open('./ratings_test.txt', encoding = 'utf-8')

test_text = []
test_features = []
test_target = []

prac = 0
for line in test_file:
	test_text.append(line)
	if prac == 10:
		break
	prac = prac + 1
	train_text.append(line)

del test_text[0]

for line in test_text:
	test_features.append(line.split('\t')[1])
	test_target.append(int(line.split('\t')[2].split('\n')[0]))

print("test_features :  ", test_features[0])
print("test_target :  ", test_target[0])

def NLP(sentences):
### konlpy로 train 문장들 형태소 분석 ###

	from konlpy.tag import Twitter

	twitter = Twitter()
	konlpy_features = []

	for sentence in sentences:
		tagged_sentence = twitter.pos(sentence)

		temp_list = []

		for word in tagged_sentence:
			if word[1] != 'Josa' and word[1] != 'Punctuation':

				temp_list.append(word[0])
		konlpy_features.append(temp_list)

	return konlpy_features

konlpy_train_features = NLP(train_features)
konlpy_test_features = NLP(test_features)

from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("w2v_2.model")
# num_features = 300

vec = np.zeros((300,), dtype="float32")
index2word_set = set(model.wv.index2word)

### [[[숫자300개],[숫자300개],[숫자300개]],[[숫자300개],[숫자300개],[숫자300개]],[[숫자300개],[숫자300개],[숫자300개]]...] ###
###     word    #    word   #   word    #    word   #    word   #    word    #    word   #    word   #    word   #    ###     
###                SENTENCE             #              SENTENCE              #              SENTENCE             #    ###

entire_sentence_vec = []

for i in konlpy_train_features:
	
	one_sentence_vec = []
	
	for word in i:

		if word in index2word_set:
			vec = np.add(vec,model[word])
		else:
			vec = np.random.randn(300)

		one_sentence_vec.append(vec)

	entire_sentence_vec.append(one_sentence_vec)



### padding 0으로  추가하기  ###

max_len = 0

for i in range(len(entire_sentence_vec)):
	if max_len < len(entire_sentence_vec[i]):
		max_len = len(entire_sentence_vec[i])

# print(max_len)

padding = np.zeros((300,), dtype="float32")

for i in range(len(entire_sentence_vec)):
	sentence_len = len(entire_sentence_vec[i])
	diff = max_len - sentence_len
	for j in range(diff):
		entire_sentence_vec[i].append(padding)

# print(entire_sentence_vec[0])        # 이거는 지금 한 문장을 말하는 것. 문제점이 뭐냐면 예시 - 논문은 벡터가 1인걸로 한 것임.
# print(type(entire_sentence_vec[0]))  # list가 나오고
# print(type(entire_sentence_vec[0][0]))  # 이거는 numpy.ndarray가 나올것임.
### CNN ###

import tensorflow as tf


embedding_size = 300
sequence_length = max_len   # 지금 여기서 max_len이 19인가 그랬음. 일단 vocab_size만 넣어주면 됨.
num_classes = 2

input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name = "input_x")   
# sequence_length는 한 문장에 단어 max 수.
input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
# 지금 논문에서는 여기에서 우리의 벡터를 집어넣는 것이다. 
# class를 이용했기 때문에 cnn.input_x: x_batch라고 했다. 
# x_batch의 형태를 알아보고 input_x에 값을 넣어주어야겠다.
# 그렇다면, x_batch의 형태를 알아내고 y_batch의 형태를 알아내면 vocab_size도 알아낼 수 있지.
# 보통 placeholder를 세션을 열고 쓰는구나. sess = tf.Session()
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# z = tf.mul(x, y)
# sess = tf.Session()
# sess.run(z, feed_dict = {x: [[3., 3.],[3.,3.]], y: [[5.,5.],[5.,5.]]})
# 15.0 이렇게 나옴.
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("filter_sizes","3,4,5","Comma-separated filter sizes (default: '3,4,5')")

vocab_size = len(index2word_set)

# # W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name = "W")
# W = tf.Variable(tf.random_uniform([vocab_size],-1.0, 1.0), name = "W")  
# print(W.get_shape())
# embedded_chars = tf.nn.embedding_lookup(W, input_x)           # W shape은 (?, ?)인데 처음은 vocab_size 그리고 embeddig size
# print(embedded_chars.get_shape())
# embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)  # embedded_chars.get_shape() = (?, ?, ?)
# print(embedded_chars_expanded.get_shape())
filter_size = 3 ## 논문 기준, github코드 기준.
embedding_size = 300  ## 우리가 300 차원으로 설정했음.
num_filters = 32      ## 128도 있긴한데 처음에는 32로 하는 듯. 모두의 딥러닝 강의 기준으로 32, 깃헙은 128 다음레이어에서 128 써보자.


pooled_outputs = []

filter_shape = [filter_size, embedding_size, 1, num_filters]
W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b")

entire_sentence_vec = np.array(entire_sentence_vec)
print(entire_sentence_vec.shape)
print(type(entire_sentence_vec[0]))
print(type(entire_sentence_vec[0][0]))
entire_sentence_vec = np.reshape(entire_sentence_vec, (7, 19, 300, 1))
print(entire_sentence_vec.shape)
# 그러면 결과적으로 filter_shape는 지금 [3, 300, 1, 32]
conv = tf.nn.conv2d(
	entire_sentence_vec.astype(np.float32),
	W,
	strides = [1,1,1,1],
	padding = "VALID",
	name = "conv")
h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")

pooled = tf.nn.max_pool(
	h,
	ksize = [1, sequence_length - filter_size + 1, 1, 1],
	strides = [1,1,1,1],
	padding = "VALID",
	name = "pool")

pooled_outputs.append(pooled)

num_classes = 2
filter_sizes = 3
num_filters_total = num_filters * 3
h_pool_flat = tf.concat(pooled_outputs, 3)

session_conf = tf.ConfigProto(
	allow_soft_placement = True,
	log_device_placement = False)
sess = tf.Session(config=session_conf)
l2_loss = tf.constant(0.0)
# num_classes = y_train.shape[1] [0,1]이 positive [1,0]이 negative로 나눴기 때문에 그냥 2로해도 됨.
num_classes = 2

filter_sizes = 3
num_filters_total = num_filters * 3
W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "b")
l2_loss += tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)

h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
scores = tf.nn.xw_plus_b(h_drop, W, b, name = "scores")

predictions = tf.argmax(scores, 1, name = "predictions")
losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
l2_reg_lambda = 0.0
loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
esv_tuple = tuple(entire_sentence_vec)



sess.run(tf.global_variables_initializer())
sess.run(loss, feed_dict = {input_x: entire_sentence_vec, input_y:[np.array([0,1])], dropout_keep_prob: 0.5})
# 그러면 우리꺼도 세션을 열고 sess.run(??, feed_dict = {input_x: ???})
# 이렇게 줘야겠네. ??? 여기 자리에 x_batch가 들어가니까 x_batch의 형태를 찾아내야한다.
