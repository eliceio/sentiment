from django.shortcuts import render
from django.http import JsonResponse
from sklearn.svm import SVC
# from sklearn.linear_model import LinearRegression as LR
import numpy as np
from gensim.models import Word2Vec
from konlpy.tag import Twitter
# Word2Vec Instance
w2v_base_model = Word2Vec.load('./demo/models/w2v_for_demo.model')

# Tiwtter Instance
tw = Twitter()

def svm(req, C=1.0):
    model = SVC(C=float(C))
    X = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,10,11],
        [31,22,13,14],
        [15,16,17,44],
        [19,20,21,22]
    ]
    y = [
        1,2,3,4,5,6
    ]
    model.fit(X, y)
    data = [
        {
            'C':C,
            'class_weight':str(model.class_weight_),
            'model_shape_fit':str(model.shape_fit_)
        } 
    ] 
    return JsonResponse(data, safe=False)

def clf(req, sentence=""):
    w2v_model = w2v_base_model
    vsize, wsize, min_count = w2v_model.vector_size, w2v_model.window, w2v_model.min_count
    max_seq_len = -1 # HAVE TO KNOW!
    sentence = sentence.strip()
    sentence_segments = sentence.split()
    # load graph
    # 
    #  
    twitter = tw
    psentence = []
    pos_tagged = twitter(sentence)
    for word, tag in pos_tagged:
        if tag in ('Punctuation', 'josa'):
             continue
        psentence.append(word)

    cnt = 0; s = []
    for word in psentence:
        try:
            s.append(w2v_model[word])
        except:
            s.append(np.zeros(vsize, dtype="float32"))
    cnt += 1
    for i in range(max_seq_len - cnt):
        s.append(np.zeros(vsize, dtype="float32"))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        
        predict, alpha = sess.run([predict, alpha], feed_dict={'X':psentence, 'Y':0, Sequence_Length:len(psentence)})

    
    return JsonResponse([
        {
            'answer':predict
        }
    ], safe=False)




def wat(req, sentence=""):
    try:
        a,b,d = sentence.strip().split()
    except:
        return JsonResponse([ { 'answer':'단어의 수가 맞지 않습니다.', 'resultcode': 0, 'passed':sentence} ], safe=False)

    validated, v_code = wat_sentence_validation((a,b,d))
    if not validated:
        if v_code == 0:
            answer = "단어의 수가 맞지 않습니다."
        elif v_code == 1:
            answer = "한글이 아닌 문자가 포함되어있습니다."
        else:
            answer = "validation failed..."
        return JsonResponse([ { 'answer': answer, 'resultcode': 0, 'description': v_code }], safe=False )
    model = w2v_base_model
    try:
        av,bv,dv = map(lambda x: model.wv[x], (a,b,d))
    except:
        return JsonResponse([ {'answer': 'word not in vocabulary', 'resultcode': 0} ], safe=False)
    
    alts = model.wv.similar_by_word(av-bv+dv) 
    word, prob = alts[0]
    
    return JsonResponse([
        {
            'answer':word,
            'resultcode': 1,
            'words': alts
        }
    ], safe=False)

def wat_sentence_validation(words_list):
    validation_fns = [isThree, isHangul]
    for i, fn in enumerate(validation_fns):
        if not fn(words_list):
            return False, i
    return True, 200

def isThree(words_list):
    if len(words_list) != 3:
        return False
    return True

def isHangul(words_list):
    for word in words_list:
        for ch in word:
            ov = ord(ch)
            if not (44032 <= ov <= 55204 or 12353 <= ov <= 12687):
                return False
    return True
