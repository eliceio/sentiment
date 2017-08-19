'''
# https://github.com/likejazz/cnn-text-classification-tf/blob/master/train.py
# 의 FLAGS를 따로 정의한 것
# evaluate_every, checkpoint_every, 
# allow_soft_placement, log_device_placement 는 뭔지 아직 모름.. 나윤
'''
FLAGS = { 
  "vec_size": 300, 
  "seq_size": 70,
  "filter_size": [3, 4, 5],
  "filter_num": 2,
  "drop_rate":0.5,
  "lambda": 0.0,
  "batch_size": 64,
  "epoch": 200,
  "evaluate_every": 100,
  "checkpoint_every": 100,
  "allow_soft_placement": True,
  "log_device_placement": False
}