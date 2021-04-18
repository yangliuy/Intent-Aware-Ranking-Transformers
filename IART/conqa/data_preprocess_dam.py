'''
Data preprocess of MS_V2 and UDC data for running DAM model
A good preprocess is very important for good performance
Preprocess UDC for debugging purpose
Preprocess MS_V2 to get results of DAM on MS_V2
The input data format is label \t context (utterances seperated by \t) \t response

Firstly run data_preprocess_dam.py, then run transfer_mz_to_dam_format.py

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@homepage: https://sites.google.com/site/lyangwww/
'''

# /bin/python2.7
import os
import sys

sys.path.append('../utils/')

from preparation import Preparation
from preprocess import Preprocess, NgramUtil

def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'please input params: data_name (udc or ms_v2)'
        exit(1)
    data_name = sys.argv[1]  # udc or ms_v2

    basedir = '../../data/' + data_name + '/'

    # transform context/response pairs into input pkl file of DAM model
    # the input files are train.txt/valid.txt/test.txt
    # the format of each line is 'label context response'
    prepare = Preparation()

    if data_name == 'udc' or data_name == 'ms_v2':
        train_file = 'train.txt'
        valid_file = 'valid.txt'
        test_file = 'test.txt'
    else:
        raise ValueError('invalid data name!')

    corpus, rels_train, rels_valid, rels_test = prepare.run_with_train_valid_test_corpus_dmn(
        basedir + train_file, basedir + valid_file,
        basedir + test_file)
    for data_part in list(['train', 'valid', 'test']):
        if data_part == 'train':
            rels = rels_train
        elif data_part == 'valid':
            rels = rels_valid
        else:
            rels = rels_test
        print 'total relations in ', data_part, len(rels)
        prepare.save_relation(basedir + 'relation_' + data_part + '.txt',
                              rels)
        print 'filter queries with duplicated doc ids...'
        prepare.check_filter_query_with_dup_doc(
            basedir + 'relation_' + data_part + '.txt')
    print 'total corpus ', len(corpus)
    prepare.save_corpus_dmn(basedir + 'corpus.txt', corpus, '\t')
    print 'preparation finished ...'

    print 'begin preprocess...'
    # Prerpocess corpus file
    # Trying not filtering terms by frequency
    preprocessor = Preprocess()
    dids, docs = preprocessor.run_2d_smn(
        basedir+ 'corpus.txt')  # docs is [corpus_size, utterance_num, max_text1_len]
    preprocessor.save_word_dict(basedir + 'word_dict.txt')
    # preprocessor.save_words_df(basedir + 'word_df.txt')

    fout = open(basedir+ 'corpus_preprocessed.txt', 'w')
    for inum, did in enumerate(dids):
        doc_txt = docs[inum]  # 2d list
        doc_string = ''
        for utt in doc_txt:
            for w in utt:
                doc_string += str(w) + ' '
            doc_string += '\t'
        fout.write('%s\t%s\t%s\n' % (
        did, len(docs[inum]), doc_string))  # id text_len text_ids
    fout.close()
    print('preprocess finished ...')