import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva


def test(conf, _model):
    
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))    
    print('finish loading data')
    print('init intent_dict...')
    conf['intent_dict'] = reader.read_intent(conf['intent_vec_path']) if conf[
                            'model_name'] != 'dam' else None
    test_batches = reader.build_batches(test_data, conf)
    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations:')
    conf_copy = {}
    for k in conf:
        if k != 'intent_dict':
            conf_copy[k] = conf[k]
    print(conf_copy)

    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    with tf.Session(graph=_graph) as sess:
        #_model.init.run();
        _model.saver.restore(sess, conf["init_model"])
        print("sucess init %s" %conf["init_model"])

        batch_index = 0
        step = 0

        score_file_path = conf['save_path'] + 'score.test'
        score_file = open(score_file_path, 'w')
        attention_file = open(conf['save_path'] + 'attention.test', 'w')

        print('starting test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for batch_index in xrange(test_batch_num):
                
            feed = { 
                _model.turns: test_batches["turns"][batch_index],
                _model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                _model.every_turn_len: test_batches["every_turn_len"][batch_index],
                _model.response: test_batches["response"][batch_index],
                _model.response_len: test_batches["response_len"][batch_index],
                _model.label: test_batches["label"][batch_index]
                }
            if conf['model_name'] != 'dam':
                feed[_model.turns_intent] = \
                    test_batches["turns_intent"][batch_index]
                feed[_model.response_intent] = \
                    test_batches["response_intent"][batch_index]
                
            scores, attention = sess.run([_model.logits, _model.attention], feed_dict = feed)
            # shape of attention [batch, max_turn_num]
            # shape of scores [batch]
            # also run and print out attention weights to do visualization
            #print('print out attention weights over context utterances:', attention)

            # print predicted scores and labels into score file
            # print intent aware-attention weights into attention file
            for i in xrange(conf["batch_size"]):
                score_file.write(
                    str(scores[i]) + '\t' + 
                    str(test_batches["label"][batch_index][i]) + '\n')
                    #str(sum(test_batches["every_turn_len"][batch_index][i]) / test_batches['tt_turns_len'][batch_index][i]) + '\t' +
                    #str(test_batches['tt_turns_len'][batch_index][i]) + '\n')
                attention_file.write('\t'.join([str(a) for a in attention[i]])
                                     + '\n')
        score_file.close()
        attention_file.close()
        print('finish test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        #write evaluation result
        result = eva.evaluate(score_file_path)
        result_file_path = conf["save_path"] + "result.test"
        with open(result_file_path, 'w') as out_file:
            for p_at in result:
                out_file.write(str(p_at) + '\n')
        print('finish evaluation')
        # lyang: also print metrics in log file
        print('testing_metrics for_model_ckpt:\t{:s}\t[current metrics (r2@1 r10@1 r10@2 r10@5 map)]\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}'.format(
            conf["init_model"], result[0], result[1], result[2], result[3], result[4]))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        

                    
