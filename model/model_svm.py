from konlpy.tag import Twitter
from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np


file = open('./ratings_train.txt', encoding="utf-8")

temp_list = []
for line in file:

	temp_list.append(line)

train_sentence_list = []
train_tag_list = []

for i in temp_list:
	if i != temp_list[0]:
		a = i.split('\t')
		train_sentence_list.append(a[1])
		train_tag_list.append(int(a[2]))

twitter = Twitter()
# twitter 객체만들기.

train_sentence_w2v = []
for i in train_sentence_list:    
	temp_bef = twitter.pos(i, norm=True, stem=True)
	temp_del_pos = []
	for j in temp_bef:
		if j[1] != 'Josa' and j[1] != 'Punctuation':  
			temp_del_pos.append(j[0])               # 조사랑 Punctuation은 필요가 없으니까 빼는거임.
	train_sentence_w2v.append(temp_del_pos)


################# 여기까지가 trainset 아래부터가 testset ##############################
file = open('./ratings_test.txt', encoding="utf-8")
temp_list = []
cutting = 0
for line in file:
	if cutting == 10:
		break
	cutting += 1
	temp_list.append(line)

test_sentence_list = []
test_tag_list = []
for i in temp_list:
	if i != temp_list[0]:
		a = i.split('\t')
		test_sentence_list.append(a[1])
		test_tag_list.append(int(a[2]))

test_sentence_w2v = []
for i in test_sentence_list:
	temp_bef = twitter.pos(i, norm = True, stem = True)
	temp_del_pos = []
	for j in temp_bef:
		if j[1] != 'Josa' and j[1] != 'Punctuation':
			temp_del_pos.append(j[0])
	test_sentence_w2v.append(temp_del_pos)

model = Word2Vec.load("w2v_2.model")
print(model)
num_features = 300

def makeFeatureVec(words, model, num_features):
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0.
	index2word_set = set(model.wv.index2word)
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])

	featureVec = np.divide(featureVec,nwords)
	if nwords == 0:
		featureVec = np.random.randn(num_features)

	return featureVec

train_sentence_vec = []
for train_i in train_sentence_w2v:
	train_sentence_vec.append(makeFeatureVec(train_i, model, num_features))

test_sentence_vec = []
for test_i in test_sentence_w2v:
	test_sentence_vec.append(makeFeatureVec(test_i, model, num_features))


clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
    probability=False, tol=0.001, cache_size=200, class_weight=None, \
    verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
clf.fit(train_sentence_vec, train_tag_list)

print(clf.score(test_sentence_vec,test_tag_list))
