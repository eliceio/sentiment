from gensim.models import Word2Vec
from konlpy.tag import Twitter
import sys

def w2v(size):
	path = 'C:\\Users\\kwk51\\Desktop\\wikiextractor\\text\\'
	text_dir = ['AA','AB','AC']
	# text_dir = ['AA','AB','AC','AD','AE','AF']
	corpus = []
	for directory in text_dir:
		for file_num in range(size):
			wiki_object = open(path + directory + '\\wiki_' + str(file_num), encoding="utf-8")

			for line in wiki_object:
				if line == '\n':
					continue
				if "doc" in line:
					continue
				corpus.append(line)
		
	twitter = Twitter()
	tokenized_list = []

	for token in corpus:
	    tokenized_list.append(twitter.morphs(token))

	model = Word2Vec(tokenized_list, size=100, window = 4, min_count=10, workers=1, iter=100, sg=1)
	# NN으로 되어있음.
	# sg = 1 skipgram 가운데 단어 양쪽 window 싸이드만큼 맞춰야되는 단어가 8개.
	# sg = 0 CBOW가 하나. 

	model.save('w2v.model')

	# print(model.most_similar(positive=['여자','왕'], negative = ['남자'], topn=30))
	# print(model.similarity('남자','여자'))

	return model.most_similar(positive=['여자', '왕'], negative = ['남자'], topn=30)

if __name__ == '__main__':
	size = int(sys.argv[1])
	print(w2v(size))