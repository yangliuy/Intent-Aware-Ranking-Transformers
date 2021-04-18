'''
Generate per-query metrics with trec_eval
to do significance test and case study later

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
'''

import os
import sys
import json
import re
from tqdm import tqdm

def fix_duplicate_doc_id(mz_prediction_file):
    '''
    Before going the next steps, we need to fix the "same doc id under the
    the same query" problem. Loop the qid under each query, if we find a
    duplicated qid, add '-' + dup_times after this did
    For example, if D811 is duplidated, we change it to D811-1, D811-2, etc.
    '''
    nd_file = mz_prediction_file + '.nd'
    dup_times = 1
    with open(mz_prediction_file) as f_in, open(nd_file, 'w') as f_out:
        cur_qid = 'init'
        cache_did_set = set()
        cache_q_lines = []
        for l in f_in:
            tokens = l.strip().split()
            if tokens[0] == cur_qid:
                # same qid
                if tokens[2] in cache_did_set:
                    # means we find a duplicate doc id
                    tokens[2] += ('-' + str(dup_times))
                    print('found dup doc_id, gen a new doc_id and qid: ', tokens[2], tokens[0])
                    dup_times += 1
                cache_did_set.add(tokens[2])
                cache_q_lines.append('\t'.join(tokens) + '\n') # tokens[2] maybe changed
            else:
                # meet a new qid
                f_out.write(''.join(cache_q_lines))
                dup_times = 1 # reset
                cache_q_lines = []
                cache_q_lines.append(l)
                cache_did_set.clear()
                cur_qid = tokens[0]
                cache_did_set.add(tokens[2])
        # the last query
        # print len(cache_q_lines), len(cache_did_set)
        if (len(cache_q_lines) != 0 and len(cache_q_lines) == len(
            cache_did_set)):
            f_out.write(''.join(cache_q_lines))
            print('write the last query... done: ', ''.join(cache_q_lines))
        return nd_file

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'please input params: mz_prediction_file (absolute path)'
        exit(1)
    mz_prediction_file = sys.argv[1] # dmn_cnn_prf_body.predict.test.txt for smn
    # dmn_cnn_kd_word-embedsize-500-rid-17-test-iter-250.predict.test.txt for ms
    # dmn_cnn_kd_word-embedsize-100-rid-17-test-iter-300.predict.test.txt for udc
    # dmn_cnn-pureDMN-contextlen-4-test-iter-500.predict.test.txt for ms

    # DMN for MS_V2   ../data/ms_v2/ModelRes/ms_v2-dmn_cnn_pure-goulburn-weights-best289-04092018-iter-290.predict.test.txt
    # DMN-PRF for MS_V2   ../data/ms_v2/ModelRes/ms_v2-dmn_cnn_prf_body-contextlen-10.weights.250.predict.test.txt
    # DMN-KD for MS_V2   ../data/ms_v2/ModelRes/ms_v2-dmn_cnn_kd_word-contextlen-6-rid-20.weights.350.predict.test.txt
    # SMN for MS_V2 ../data/ms_v2/ModelRes/ms_v2-smn-test.pkl.pred.txt.mz-score-file

    # Before going the next steps, we need to fix the "same doc id under the
    # the same query" problem. Loop the qid under each query, if we find a
    # duplicated qid, add '-' + dup_times after this did
    # For example, if D811 is duplidated, we change it to D811-1, D811-2, etc.
    mz_prediction_file = fix_duplicate_doc_id(mz_prediction_file)

    # Q597901	Q0	D118777	0	2.370282	DMN_CNN	0(ground truth)
    # seperated by \t
    # qid \t Q0 \t did \t rank \t score \t method \t ground_truth_label
    # 030  Q0  ZF08-175-870  0   4238   prise1
    # qid iter   docno      rank  sim   run_id
    # In particular, note that the rank field is ignored here;
    # internally ranks are assigned by sorting by the sim field with ties
    # broken deterministicly (using docno).
    with open(mz_prediction_file) as f_in, open(mz_prediction_file + '.score', 'w') as score_out, open(mz_prediction_file + '.qrel', 'w') as qrel_out:
        for l in f_in:
            to = l.split('\t')
            score_out.write(' '.join(to[0:len(to)-1]) + '\n')
            qrel_out.write(to[0] + ' Q0 ' + to[2] + ' ' + to[6]) # qid  iter  docno  rel

    # compute per-query metrics with qrel
    # use -q to print out metrics for all queries
    cmd = '''trec_eval -m 'all_trec' -q ''' + mz_prediction_file + '.qrel ' + mz_prediction_file + '.score > ' \
          + mz_prediction_file + '.metrics'
    print 'run ', cmd
    os.system(cmd)

    # parse the metrics file to extract the used metrics into a json file
    # {'Q101' : {'map':0.876, 'recall5':0.876, 'recall1':0.876, 'recall2':0.876}}
    q_metrics_dict = {}
    used_metrics = {'map', 'recall_1', 'recall_2', 'recall_5'}
    with open(mz_prediction_file + '.metrics') as f_in:
        #'\s+' to match 1 to many spaces
        for l in tqdm(f_in):
            to = re.split('\s+', l) # m_name qid score
            if to[0] not in used_metrics:
                continue
            if to[1] in q_metrics_dict:
                q_metrics_dict[to[1]][to[0]] = float(to[2])
            else:
                q_metrics_dict[to[1]] = {}
                q_metrics_dict[to[1]][to[0]] = float(to[2])
    with open(mz_prediction_file + '.metrics.json', 'w') as outfile:
        json.dump(q_metrics_dict, outfile)


