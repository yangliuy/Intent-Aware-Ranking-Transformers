'''
A scripts to help run gen_query_mz_score_file_from_dam_scores.py and
gen_query_all_metrics.py
'''


import os

dam_score_dict = {
    'ms_v2-iart-att-dot': '../../output/ms_v2/iadam-attention-iadam-att-dot-opd-test-dot-run40/score.test',
    'ms_v2-iart-att-outprod': '../../output/ms_v2/iadam-attention-iadam-att-dot-opd-test-outprod-run40/score.test',
    'udc-iart-att-dot': '../../output/udc/iadam-attention-iadam-att-intentv2-test-dot-intentv2-run44/score.test',
    'udc-iart-att-outprod': '../../output/udc/iadam-attention-iadam-att-intentv2-test-outprod-intentv2-run44/score.test'
}

for data in ['ms_v2', 'udc']:
    for model in ['iart-att-dot', 'iart-att-outprod']:
        dam_prediction_score_file = dam_score_dict[data + '-' + model]
        cmd = 'python gen_query_mz_score_file_from_dam_scores.py ' + \
              dam_prediction_score_file + ' ' + data
        print 'run ', cmd
        os.system(cmd)
        mz_prediction_file = dam_prediction_score_file + '.mz_score'
        cmd = 'python gen_query_all_metrics.py ' + mz_prediction_file
        print 'run ', cmd
        os.system(cmd)