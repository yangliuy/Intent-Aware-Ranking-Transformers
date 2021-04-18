import  pickle
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'please input params: word_embedding pkl file'
        exit(1)
    word_embed_file = sys.argv[1]
    word_embedding_init = np.array(pickle.load(open(word_embed_file, 'rb')))
    print('shape of word_embedding_init: ', word_embedding_init.shape)
    print('init embed vectors of the first 10 words are: ', word_embedding_init[0:10,:])

