'''
# 2017. 08. 18. dopha-mipa
# word embedding layer

# def tokenizing_sentence():  문장을 konlpy로 전처리 후 반환
# def token_to_2d(sentence):  문장에 포함된 토큰의 벡터를 CNN input 형태로 반환
# def already_tokenized():    konlpy를 이미 마친 데이터 파일을 불러와 반환

# TODO : 리뷰가 비는 경우가 생긴다... 무시하도록 짜고 나중에 해결
'''
import numpy as np
from konlpy.tag import Twitter
from gensim.models import Word2Vec
from cnn_global import FLAGS

# def main():
  # train_token, train_label = tokenizing_sentence(
                                # "data/senti/ratings_train.txt")

  # tokenized = already_tokenized()
  # 토큰화된 문장 하나를 최대길이에 맞춰 CNN input으로 바꿔줌
  # mat = token_to_2d(tokenized[0])

'''
# 문장(리뷰)별 token들을 cnn input (2d) 로 변환 
'''
def token_to_2d(sentence):
  model = Word2Vec.load("w2v_senti.model")
  list_2_mat = []
  vec = [0 for x in range(FLAGS["vec_size"])]

  for token in sentence:
    if token in model.wv.vocab:
      vec = list(model[token])
    
    list_2_mat.append(vec)

  while len(list_2_mat) < FLAGS["seq_size"]:
    list_2_mat.append([0 for x in range(FLAGS["vec_size"])])

  list_2_mat = np.asarray(list_2_mat)
  return list_2_mat

'''
# TODO : filename으로 호출할 것.
# 데이터를 한국어 전처리 (konlpy) --- 
'''
def tokenizing_sentence(filename):
  file_t = open(filename, "rt", encoding="UTF-8")
  # 첫 줄은 id, document, label
  line_train = file_t.readline()
  line_train = file_t.readline()

  twitter = Twitter()
  tokenized_sentence = []
  label = []

  '''
  # 1. 자연어 정제 ---- 추후 RNN 팀과 맞춰야 할 수 있음! TODO
  # 데이터의 sentence에 대해 konlpy를 이용한 tokenizing
  '''
  num_for_print = 0
  while line_train != "":
    # 리뷰 전처리 - id 제거
    sentence = line_train.strip()
    space_idx = sentence.find(' ')
    sentence = sentence[space_idx:]

    temp_bef = twitter.pos(sentence[:-1], norm=True, stem=True)
    temp_del_pos = list()
    for j in temp_bef:
      if j[1] != 'Josa' and j[1] != 'Punctuation':
        temp_del_pos.append(j[0])

    if (num_for_print % 10000) == 0:
      print(num_for_print)

    num_for_print += 1
    label.append(sentence[-1])  # 라벨 추가
    tokenized_sentence.append(temp_del_pos)  # 처리된 리뷰 추가

    line_train = file_t.readline()

  file_t.close()
  return tokenized_sentence, label

def already_tokenized():
  # 이미 전처리된 데이터 (감정 데이터)를 konlpy 없이 로딩합니다.
  file_token = open("tokenized_senti.txt", "rt", encoding="UTF-8")
  line = file_token.readline()
  tokenized = []

  num_iter = 0
  max_len = 0
  while line != "":
    row = line.strip().split(", ")
    tokenized.append(row)
    line = file_token.readline()

    # if max_len < len(row):
    #   max_len = len(row)
    if num_iter % 50000 == 0:
      print(num_iter)
    num_iter += 1

  file_token.close()
  return tokenized

if __name__ == "__main__":
  main()