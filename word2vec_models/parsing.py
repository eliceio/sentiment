from pickle import (
    dump as pdump
)
from konlpy.tag import Twitter
tw = Twitter()
pos = tw.pos



def load_files():
    wiki = open("./text_data/wiki_space_tokenizer.txt", encoding="utf-8")
    naver_movie = open("./text_data/naver_movie.txt", encoding="utf-8")
    return [wiki, naver_movie]

def naver_parsing(corpus):
    corpus.readline() # skip header
    sentences = []
    tags = []
    for line in corpus:
        if len(line) <= 1:
            continue
        rid, raw_sentence, tag = line.split('\t')
        parsed = pos(raw_sentence)
        if not parsed:
            continue
        sentence = []
        for word in parsed:
            w, p = word
            if p not in ('Punctuation', 'Josa') and len(w) > 1:
                sentence.append(w)
        sentences.append(sentence)
        tags.append(tag)
    corpus.close()
    return sentences, tags

def wiki_parsing(corpus):
    sentences = []
    for line in corpus:
        if len(line) <= 1:
            continue
        parsed = pos(line)
        if not parsed:
            continue
        sentence = []
        for word in parsed:
            w, p = word
            if p not in ('Punctuation', 'Josa') and len(w) > 1:
                sentence.append(w)
        sentences.append(sentence)
    corpus.close()
    return sentences

def save_dump(sentences, file_name):
    try:
        with open(file_name, 'wb') as fp:
            pdump(sentences, fp)
        print("sentence binary pickle generated!")

    except:
        print("Error!")

def run():
    wiki, naver_movie = load_files()
    
    print("Parsisng Wiki...")
    sentences1 = wiki_parsing(wiki)
    print("ENDS!")
    print("wiki sentence Dumping")    
    save_dump(sentences1, 'wiki_sentences.bin')
    print("ENDS")
    #print("Parsing Naver movie...")
    #sentences2, tags = naver_parsing(naver_movie)
    #print("ENDS")
    #print("Naver Movie sentence and tag DUMPING")
    #save_dump(sentences2, 'naver_movies_sentences.bin')
    #save_dump(tags, "tags.bin")
    #print("ENDS!")
    #print("Making all DUMPS!")
    #save_dump(sentences1.extend(sentences2), "sentences.bin")
    #print("ENDS")
if __name__ == '__main__':
    run()  


