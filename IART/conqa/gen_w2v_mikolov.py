'''
Train word embeddings with word2vec tool by mikolov with training data of ms/udc
The script for generate the pretrained word embedding file for DAM model
We directly used the released dataset from sequential-matching-network (Wu et
al., 2017), thus there is no preprocessing phase in our experiments.

We pre-train word-embeddings using a word2vec toolkit in c++, I can hardly
find the script we used to pre-train word-embeddings. Maybe the following
command can help you:

./bin/word2vec -train $train_dat -output "$train_dat.w2v" -debug 2 -size 200 \
-window 10 -sample 1e-4 -negative 25 -hs 0 -binary 1 -cbow 1 -min-count 1

# default setting cut_embed_mikolov_200d_no_readvocab.txt
@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
'''

import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    word2vec_path = '/net/home/lyang/PycharmProjects/NLPIRNNMatchZooQA/src-match-zoo-lyang-dev/data/udc/ModelInput/word2vec_mikolov/word2vec/bin/'

    if len(sys.argv) < 3:
        print 'please input params: data_name (udc or ms_v2), model_input_folder (folder for corpus.txt)'
        exit(1)
    data_name = sys.argv[1]  # udc or ms_v2
    model_input_folder = sys.argv[2] # model_input_folder
    corpus_file = model_input_folder + 'corpus.txt'
    corpus_text_file = model_input_folder + 'corpus_text.txt'
    vocab_file = model_input_folder + 'word_dict.txt'

    print 'generate corpus_file: ', corpus_text_file
    # generate corpus_text.txt for training the word vectors by word2vec
    with open(corpus_file) as f_in, open(corpus_text_file, 'w') as f_out:
        for l in tqdm(f_in):
            #print 'l: ', l
            f_out.write(' '.join(l.split()[1:]))

    for wd in list([200,300]):
        word_embed_file = 'train_word2vec_mikolov_' + str(wd) + 'd.txt'
        cmd = word2vec_path + 'word2vec -train ' + corpus_text_file + \
              ' -output ' + model_input_folder + word_embed_file + ' -debug 2 ' \
              '-size ' + str(wd) + ' -window 10 -sample 1e-4 -negative 25 -hs 0 -binary 0 -cbow 1 -min-count 1 ' \
                                    '-threads 5'
        print 'run cmd: ', cmd
        os.system(cmd)

