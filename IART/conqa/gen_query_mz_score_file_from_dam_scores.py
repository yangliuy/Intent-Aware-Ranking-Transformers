'''
Transfer the score files of DAM related models to the format
of MatchZoo score files
The required score file format is as follows:
# Q597901	Q0	D118777	0	2.370282	DMN_CNN	0(ground truth)
# seperated by \t
# qid \t Q0 \t did \t rank \t score \t method \t ground_truth_label
# 030  Q0  ZF08-175-870  0   4238   prise1
# qid iter   docno      rank  sim   run_id
# In particular, note that the rank field is ignored here;
# internally ranks are assigned by sorting by the sim field with ties
# broken deterministicly (using docno).

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
'''

import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'please input params: dam_prediction_score_file (absolute path) data_name (udc or ms_v2)'
        exit(1)
    dam_score_file = sys.argv[1]  # path of dam score file
    # format of dam score file: score label
    # the order is the same with the input data
    data_name = sys.argv[2]
    relation_file = '../../data/' + data_name + '/relation_test.txt' # use non_fd version to be consistent
    with open(dam_score_file) as fin_score, open(relation_file) as fin_rel, \
        open(dam_score_file + '.mz_score', 'w') as fout_score:
        for dam_score_line in fin_score:
            dam_score_line = dam_score_line.strip().split()
            rel_line = fin_rel.readline().strip().split()
            label = rel_line[0]
            qid = rel_line[1]
            did = rel_line[2]
            score = dam_score_line[0]
            fout_score.write(qid + '\tQ0\t' + did + '\t0\t' + score + '\t' +
                             dam_score_file + '\t' + label + '\n')

