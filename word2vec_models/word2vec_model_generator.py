from konlpy.tag import Twitter
from gensim.models import Word2Vec
from pickle import load as pload
mcnt_min = 10
mcnt_max = 50
cnt_step = 10
min_vsize = 50
max_vsize = 301
step = 25
min_wsize = 10
max_wsize = 11



log = open("./log.txt", 'w+', encoding="utf-8")

try:
    with open('./dumps/sentences_final.bin', 'rb') as fp:
        sentences = pload(fp)
except:
    print('load failed!')
    exit()
    

# Word2Vec
print("Models are now being generated!")
for min_cnt in range(mcnt_min, mcnt_max, cnt_step):
    for size in range(min_vsize, max_vsize, step):
        for windowSize in range(min_wsize, max_wsize):
            try:
                filename = "./models/w2v_window%d_size%d_mincount%d.model" % (windowSize, size,min_cnt)
                Word2Vec(sentences, size=size, window=windowSize, min_count=min_cnt).save(filename)
                print('model(window=%d, vectorsize=%d) generated!' % (windowSize, size))
                
                log.write("model(window=%d, vectorsize=%d, mincount=%d) is generated\n" % (windowSize,size,min_cnt))
            except:
                log.write("model(window=%d, vectorsize=%d, mincount=%d) is not available\n" % (windowSize,size,min_cnt))

log.close()
# Glove
    
