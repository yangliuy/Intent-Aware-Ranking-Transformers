'''
Filter word embddings from pre-trained word embeddings from Mikolov tool
word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
there is one dimension in vocab_size which is corresponding to _eos_.
in our preprocessing, _eos_ is always the last dimension
+1 to add one more embedding vector for padding and masking
We add an "all 0" vector in the 0-th row of word_embedding_init in order
to denote the padding word
when call tf.nn.embedding_lookup(), if word_id = 0, then this is a paded
word; if word_id > 0 (from 1 to vocab_size), then this is a real word

Can double check the relationships between the row index and word ids
@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@homepage: https://sites.google.com/site/lyangwww/
'''

import sys
import numpy as np
import pickle
from tqdm import tqdm



w2v_file = open(sys.argv[1]) # pre-trained word embedding file by the mikolov tool
word_dict_file = open(sys.argv[2]) # word_dict file for the vocabulary information
output_file = open(sys.argv[3], 'wb') # the cutted word embedding file with the shape [vocab_size+1, emb_size]

# In the generated word_embedding.pkl file
# the 0-th row if a "all 0" vector which is corresponding to the padded word
# the 1-vocab_size -th row are word vectors for real words in vocabs

# word_count, embed_dim = w2v_file.readline().strip().split()

word_map_w2v = {}
word2id = {} # map word to id
id2word = {} # map id to word

print 'load word dict ...'
for line in tqdm(word_dict_file):
  line = line.split()
  try:
      word2id[line[0]] = int(line[1])
      id2word[int(line[1])] = line[0]
  except:
      print line
      continue

# add one word for _eos_ and _pad_
# pad_id = 0
eos_id = len(word2id) + 1

# word2id['_pad_'] = pad_id
word2id['_eos_'] = eos_id
# id2word[pad_id] = '_pad_'
id2word[eos_id] = '_eos_'
vocab_size = len(word2id)

print 'vocab_size = ', vocab_size

print 'load word vectors ...'
for line in tqdm(w2v_file):
  line = line.split()
  if len(line) == 0 or len(line) == 2: # len(len) == 2 for the first line (V WD)
      continue
  if line[0] in word2id:
    word_map_w2v[line[0]] = line[1:]

emb_size = len(word_map_w2v[word_map_w2v.keys()[0]])

print 'emb_size = ', emb_size

word_diff = list()
for w in word2id.keys():
    if w not in word_map_w2v:
        word_diff.append(w)

# output shared w2v dict
word_embedding_init = np.array(np.zeros((vocab_size+1, emb_size))) # a 2-d array with shape [vocab_size+1, emb_size]

print 'number of shared word vectors: ', vocab_size-len(word_diff)

# the first row is an all 0 vector for padding word

# then add init embedding vectors for real words

for id in tqdm(range(1, vocab_size+1)):
    word = id2word[id]
    if word in word_map_w2v:
        word_embedding_init[id,:] = [float(s) for s in word_map_w2v[word]]
    else:
        alpha = 0.5 * (2.0 * np.random.random() - 1.0)
        rand_embed = (2.0 * np.random.random_sample([emb_size]) - 1.0) * alpha
        rand_embed = ['%.6f' % k for k in rand_embed.tolist()]
        word_embedding_init[id, :] = rand_embed

pickle.dump(word_embedding_init.tolist(), output_file)

print 'Map word vectors finished ...'
