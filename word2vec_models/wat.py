from gensim.models import Word2Vec
min_cnt_min = 10
min_cnt_max = 50
min_cnt_step = 10
wsize_min = 2
wsize_max = 10
vsize_min = 50
vsize_max = 300
vsize_step = 25

# n_wats = 14
n_wats = 2
wats_list = []
for i in range(n_wats):
    words_list = []
    with open('./wats/wat%d' % i, 'r', encoding='utf-8') as f:
        for line in f:
            words_list.append(line.strip().split())
    wats_list.append(words_list)

print(wats_list[1])
             
logger = open('wat_log.txt', 'w+')
def calcAccuracy(path):
    print(path)
    w2v_model = Word2Vec.load(path)
    scores = []
    tot_score = 0
    for idx, wat in enumerate(wats_list):
        score = 0
        for words_list in wat:
            try:
                a,b,c,d = words_list
            except:
                continue
            a,b,c,d = map(lambda x:x.strip(), (a,b,c,d))
            try:
                av, bv, cv, dv = map(lambda x: w2v_model[x], (a,b,c,d))
            except:
                continue
            alts = w2v_model.wv.similar_by_vector(av - bv + dv)
            idx = -1
            for i in range(10):
                vocab, prob = alts[i]
                if vocab == c:
                    idx = i
                    break
            if idx == -1:
                continue
            score += (1 / float(idx + 1))
            tot_score += (1 / float(idx + 1))
        scores.append(score / float(len(wat)))
        print(i, score, len(wat))
    print("total hits score : ", tot_score / sum(len(w) for w in wat))
    return scores
        
def run():

    logger = open('wat_log.txt', 'w+')

    for mcnt in range(min_cnt_min, min_cnt_max+1, min_cnt_step):
        for wsize in range(wsize_min, wsize_max+1):
            for vsize in range(vsize_min, vsize_max+1, vsize_step):
                modelPath = "./models/w2v_window%d_size%d_mincount%d.model" % (wsize, vsize, mcnt)
                scores = calcAccuracy(modelPath)
                logger.write('wsize:%d, vsize:%d, mincount:%d\n' % (wsize, vsize, mcnt))
                for i, score in enumerate(scores):
                    logger.write('task %d: %.4f\n' % (i, score))         
   
    logger.close()             


if __name__ == '__main__':
    run()


