import pickle
FNAMEFORMAT = "class%d"



def run():
    dumps = []
    cnames = []
    idx = 0
    with open('./checked.txt', encoding="utf-8") as f:
        chunk = []
        for line in f:
            if ':' in line:
                cnames.append(line)
                if idx != 0:
                    dumps.append(chunk)
                    chunk = []
      
                idx += 1
                continue
            chunk.append(line)

    for i in range(len(dumps)):
        if not dumps[i]:
            continue
        fname = "class%d" % i
        with open(fname, 'wb') as fp:
            pickle.dump(dumps[i], fp)
    with open("wat_classes", 'wb') as fp:
        pickle.dump(cnames, fp)
    
    
            
            
     

if __name__ == '__main__':
    run()
