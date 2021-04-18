'''
Data preprocess of MS_V2 and UDC data for running DAM model
Transfer preprocessed data in MZ format to DAM input format
A good preprocess is very important for good performance
Preprocess UDC for debugging purpose
Preprocess MS_V2 to get results of DAM on MS_V2
The input data format is label \t context (utterances seperated by \t) \t response
Add qid/dids in relation files in the pkl file in order to associate the
corresponding intent vectors in the future

Firstly run data_preprocess_dam.py, then run transfer_mz_to_dam_format.py

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@homepage: https://sites.google.com/site/lyangwww/
'''

import sys
import pickle
import random
from tqdm import tqdm

def gen_id2corpus(corpus_pre_file, word_dict_file):
    word_dict = dict()
    id2corpus = dict()
    id2word = dict()
    with open(word_dict_file, 'r') as fin:
        for l in fin:
            tok = l.split()
            word_dict[tok[0]] = tok[1].strip()
            id2word[int(tok[1].strip())] = tok[0]
        # word_id for _eos_ is the maxID+1 in word_dict
        word_dict['_eos_'] = str(len(word_dict) + 1)
        id2word[len(id2word) + 1] = '_eos_'
    print('word id for _eos_: ', word_dict['_eos_'])
    #print('len(id2word)', len(id2word))
    with open(corpus_pre_file, 'r') as fin:
        for l in tqdm(fin):
            tok = l.split('\t')
            id = tok[0]
            if 'D' in id:
                id2corpus[id] = [int(w) for w in tok[2].split()]
            else:
                utts = []
                for i in range(2, len(tok)):
                    utts.extend(tok[i].split())
                    utts.append(word_dict['_eos_'])
                utts = utts[0:len(utts)-2] # remove the last 2 _eos_
                utts = [int(w) for w in utts]
                # utts_words = [id2word[w] for w in utts]
                # print('test utts_words: ', utts_words)
                id2corpus[id] = utts
    return id2corpus, word_dict

def gen_dam_inputs(basedir, data_partition, id2corpus, word_dict, gen_mode,
                   relation_mode, ag_mode, ag_sample_num):
    if relation_mode == 'nofd' \
        or data_partition == 'test' \
        or data_partition == 'valid': # can't filter queries in test/valid
        rel_file = basedir + 'relation_' + data_partition + '.txt' # for data_nofd.pkl
    elif ag_mode == 'yes':
        rel_file = basedir + 'relation_' + data_partition + '.txt.ag' + ag_sample_num   # for data_ag2.pkl
    else:
        rel_file = basedir + 'relation_' + data_partition + '.txt.fd'  # for data.pkl
    print 'using relation file: ', rel_file

    labels = []
    qids = []
    dids = []
    context = []
    resp = []

    with open(rel_file) as fin:
        ins_num = 0
        for l in tqdm(fin):
            ins_num += 1
            tok = l.strip().split()
            labels.append(int(tok[0]))
            qids.append(tok[1].strip())
            dids.append(tok[2].strip())
            context.append(id2corpus[tok[1]])
            resp.append(id2corpus[tok[2]])
            if gen_mode == 'small' and ins_num >= 10000:
                break
    # Add qid/dids in relation files in the pkl file in order to associate the
    # corresponding intent vectors in the future
    return {'y': labels, 'c': context, 'r': resp, 'qids': qids, 'dids': dids}

def gen_relation_train_ag_file(basedir, data_name, ag_sample_num):
    '''
    Perform data augumentation for training data by ramdonly sampling more
    negtive training data
    Only need to do this for UDC data
    '''
    if data_name != 'udc':
        raise NameError('can only do ag for udc!')
    rel_train_file = basedir + 'relation_train.txt'
    doc_id_pool = set()
    with open(rel_train_file) as fin:
        for l in fin:
            t = l.strip().split()
            doc_id_pool.add(t[2])
    # for UDC, each qid has 1 pos did and 1 neg did
    # we further sample k more neg dids for each qid
    doc_id_pool = list(doc_id_pool)
    total_doc_num = len(doc_id_pool)
    print 'test total_doc_num', total_doc_num
    with open(rel_train_file) as fin, open(
        rel_train_file + '.ag' + ag_sample_num, 'w') as fout:
        line_index = -1
        for l in tqdm(fin):
            t = l.strip().split()
            fout.write(l)
            line_index += 1
            if line_index % 2 == 1:
                #print 'test cur line_index and t', line_index, t
                sampled_num = 0
                while sampled_num < int(ag_sample_num): # transfer to int!
                    #print 'test sampled_num and ag_sample_num: ', sampled_num, ag_sample_num
                    pick = random.randint(0,total_doc_num-1)
                    sdid = doc_id_pool[pick]
                    #print 'test pick sdid t[2]: ', pick, sdid, t[2]
                    if sdid != t[2]:
                        fout.write('0 ' + t[1] + ' ' + sdid + '\n')
                        #print 'test sampled_num: ', sampled_num
                        sampled_num += 1

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print 'please input params: data_name (udc or ms_v2) \
        gen_mode (full or small) relation_mode(fd or nofd) ag_mode(yes or no) \
              ag_sample_num (2,4,6,8)'
        exit(1)
    # If gen_mode=small, only use 10000 train/valid/test relations for debug
    # If relation_mode=nofd, use original relation files without filtering
    # queries with duplicate doc id
    # If ag_mode=yes, do data augumentation for training data by sample
    # new negative dids from the doc id pool. If the sampled did is already
    # covered, do resampling; otherwise add this sampled <q,d> pair into
    # the training data to get a larger training data
    # transfer_mz_to_dam_format.py udc full fd yes 2
    data_name = sys.argv[1] # udc or ms_v2
    gen_mode = sys.argv[2] # full or small
    relation_mode = sys.argv[3] # fd or nofd
    ag_mode = sys.argv[4] # yes or no
    ag_sample_num = sys.argv[5] # 2,4,6,8
    basedir = '../../data/' + data_name + '/'

    corpus_pre_file = basedir + 'corpus_preprocessed.txt'
    word_dict_file = basedir + 'word_dict.txt'
    id2corpus, word_dict = gen_id2corpus(corpus_pre_file, word_dict_file)

    if ag_mode == 'yes':
        gen_relation_train_ag_file(basedir, data_name, ag_sample_num)

    # transform context/response pairs into input pkl file of DAM model
    train = gen_dam_inputs(basedir, 'train', id2corpus, word_dict, gen_mode,
                           relation_mode, ag_mode, ag_sample_num)
    valid = gen_dam_inputs(basedir, 'valid', id2corpus, word_dict, gen_mode,
                           relation_mode, 'no', ag_sample_num) # no ag for valid
    test = gen_dam_inputs(basedir, 'test', id2corpus, word_dict, gen_mode,
                          relation_mode, 'no', ag_sample_num) # no ag for test

    if gen_mode == 'small':
        data_pkl_name = 'data_small.pkl'
    elif relation_mode == 'nofd':
        data_pkl_name = 'data_nofd.pkl'
    elif ag_mode == 'yes':
        data_pkl_name = 'data_ag' + ag_sample_num + '.pkl'
    else:
        data_pkl_name = 'data.pkl'
    print('begin writing data pkl file...', data_pkl_name)
    pickle.dump((train,valid,test), open(basedir + data_pkl_name, 'wb'))
    print('finish writing data pkl file...', data_pkl_name)
    with open(basedir + 'word2id', 'w') as fout:
        for w in word_dict:
            fout.write(w + '\n')
            fout.write(word_dict[w] + '\n')
        print('write word_dict done!')





