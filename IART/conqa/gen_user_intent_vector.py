'''
Extract the predicted user intent vectors predicted by Chen Qu
Build a dict for user intent vectors
The key is qid_uid or rid (e.g. Q1-0 Q1-1,..., D0, D1, D2,...)
        since the context is 2D texts; the response is 1D text
The value is a 12-dimensional intent vector for this context utterance or
response candidate

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@created on 02/08/2019
'''

import sys
import json
import numpy as np
from tqdm import tqdm
import base64


def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict


if __name__ == '__main__':
    # Extract the 12-dimensional user intent vectors predicted by Chen Qu's
    # classifier
    if len(sys.argv) < 2:
        print ('please input params: data_name (ms_v2 or udc)')
        exit(1)
    data_name = sys.argv[1]  # udc or ms_v2

    basedir = '../../data/' + data_name + '/ModelInput/'
    cur_data_dir = basedir + 'dmn_model_input/'
    data_name_qc = data_name if data_name != 'ms_v2' else 'ms'
    # intent_file_folder = '/mnt/scratch/chenqu/response_intent/output/'
    # /mnt/scratch/chenqu/response_intent/udc_v2/output/
    intent_file_folder = '/mnt/scratch/chenqu/response_intent/udc_v2/output/'

    # Extract user intent vectors for each (context_qid, utt_index, candidate_response_id) triple
    # in train/valid/test data
    intent_dict = {}
    for data_part in list(['test', 'train', 'valid']):
        # ! Note that here we need to read non-fd version of relation files to be consistant
        # ! fd version just filtered queries with duplicated doc ids
        # ! the filtering process won't change the qids/ dids
        # ! thus the keys of qid/did can be used in both filtered version and
        # non-filtered version
        relation_file = cur_data_dir + 'relation_' + data_part + '.txt'
        intent_file = intent_file_folder + data_name_qc + '_' + data_part + '.txt'
        with open(relation_file) as fin_relation, open(
            intent_file) as fin_intent:
            print('preprcess file: ', intent_file)
            for rel_line in tqdm(fin_relation):
                intent_line = fin_intent.readline()
                intent_tokens = intent_line.split('\t')
                rel_tokens = rel_line.split()
                qid, rid = rel_tokens[1], rel_tokens[2]
                # collect intent vectors for context utterances
                for i in range(1, len(intent_tokens) - 1):
                    intent_dict[qid + '-' + str(i - 1)] = intent_tokens[
                        i].strip()
                # collect intent vectors for response candidates
                intent_dict[rid] = intent_tokens[
                    len(intent_tokens) - 1].strip()
    print('test len of intent_dict: ', len(intent_dict))
    # output to file
    intent_file = cur_data_dir + 'intent_vectors_v2.txt'
    with open(intent_file, 'w') as fout:
        for id in intent_dict:
            fout.write(id + '\t' + intent_dict[id] + '\n')
    print('intent_dict[0:10]: ', dict(intent_dict.items()[0:10]))
    print('done!')



















