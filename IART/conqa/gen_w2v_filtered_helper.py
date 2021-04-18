import sys
import os

for data in list(['ms_v2']):
      for wd in list([200, 300]):
        cur_folder = '../../data/' + data +'/'
        cmd = 'python gen_w2v_filtered.py ' \
              + cur_folder + 'train_word2vec_mikolov_' + str(wd) +'d.txt ' \
              + cur_folder + 'word_dict.txt ' \
              + cur_folder + 'cut_embed_mikolov_' + str(wd) + 'd.pkl'
        print cmd
        os.system(cmd)