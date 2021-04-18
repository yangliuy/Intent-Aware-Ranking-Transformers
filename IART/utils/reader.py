import cPickle as pickle
import numpy as np
from tqdm import tqdm

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c = np.array(data['c'])
    r = np.array(data['r'])
    qids = []
    dids = []

    assert len(y) == len(c) == len(r) == len(data['qids']) == len(data['dids'])
    p = np.random.permutation(len(y))
    # shuffle qids and dids
    for i in range(len(y)):
        qids.append(data['qids'][p[i]])
        dids.append(data['dids'][p[i]])

    shuffle_data = {'y': y[p], 'c': c[p], 'r': r[p], 'qids': qids, 'dids': dids}
    # print('test after shuffle: ')
    # print('y: ', y[p][0:1])
    # print('c: ', c[p][0:1])
    # print('r: ', r[p][0:1])
    # print('qids: ', qids[0:1])
    # print('dids: ', dids[0:1])

    return shuffle_data

def split_c(c, split_id):
    '''c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns

def normalize_length(_list, length, cut_type='tail'):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    # if real_length == 0, pad 0
    if real_length == 0:
        return [0]*length, 0

    # if real_length <= length, pad 0s
    if real_length <= length:
        # 1D list
        if not isinstance(_list[0], list):
            _list.extend([0]*(length - real_length))
        else: # 2D list
            _list.extend([[]]*(length - real_length))
        return _list, real_length

    # if real_length > length, cut extra tokens
    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length

def produce_intent(cid, rid, turns_len_all, intent_dict):
    r_intent = intent_dict[rid]
    c_intent = []
    for i in range(turns_len_all): # loop all turns in context
        c_intent.append(intent_dict[cid + '-' + str(i)])
    return c_intent, r_intent

def produce_one_sample(data, conf, index, split_id, max_turn_num, max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
    '''
    # print('keys of data: ', data.keys())
    c = data['c'][index]
    r = data['r'][index][:]
    y = data['y'][index]
    cid = data['qids'][index]
    rid = data['dids'][index]
    c_intent = []
    r_intent = []

    turns = split_c(c, split_id)
    turns_len_all = len(turns) # all turns in context before normalization

    if conf['model_name'] != 'dam':
        c_intent, r_intent = produce_intent(cid, rid, turns_len_all, conf['intent_dict'])

    #print('test c_intent: ', c_intent)
    #normalize turns_c length, nor_turns length is max_turn_num
    # cut extra conversation turns
    nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)
    if conf['model_name'] != 'dam':
        nor_turns_intent, turn_len_intent = normalize_length(c_intent, max_turn_num, turn_cut_type)
    # print('test nor_turns_intent, turn_len_intent: ', nor_turns_intent,
    #       turn_len_intent)

    nor_turns_nor_c = []
    term_len = []
    #nor_turn_nor_c length is max_turn_num, element of a list length is max_turn_len
    # cut extra length for context turn text
    for c in nor_turns:
        #nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_turns_intent_nor_it = []
    # pad 0s in nor_turns_intent if there are less than max_turn_num turns
    if conf['model_name'] != 'dam':
        for it in nor_turns_intent:
            # nor_it length is intent_size
            nor_it, nor_it_len = normalize_length(it, conf['intent_size'], term_cut_type)
            nor_turns_intent_nor_it.append(nor_it)

    # cut extra length for response text
    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len, nor_turns_intent_nor_it, r_intent

def build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns = []
    _tt_turns_len = []
    _every_turn_len = []
    _turns_intent = []

    _response = []
    _response_len = []
    _response_intent = []

    _label = []

    for i in range(conf['batch_size']):
        # i is to loop instances in the current batch
        # index is a global position for this instance
        index = batch_index * conf['batch_size'] + i
        y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len, c_intent, r_intent = produce_one_sample(data, conf, index, conf['_EOS_'], conf['max_turn_num'],
                conf['max_turn_len'], turn_cut_type, term_cut_type)

        _label.append(y)
        _turns.append(nor_turns_nor_c)
        _response.append(nor_r)
        _every_turn_len.append(term_len)
        _tt_turns_len.append(turn_len)
        _response_len.append(r_len)
        if conf['model_name'] != 'dam':
            _turns_intent.append(c_intent)
            _response_intent.append(r_intent)

    return _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label, _turns_intent, _response_intent

def build_one_batch_dict(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label, _turns_intent, _response_intent = build_one_batch(data, batch_index, conf, turn_cut_type, term_cut_type)
    ans = {'turns': _turns,
            'tt_turns_len': _tt_turns_len,
            'every_turn_len': _every_turn_len,
            'response': _response,
            'response_len': _response_len,
            'label': _label, "turns_intent": _turns_intent,
           "response_intent": _response_intent}
    return ans

def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail'):
    '''Build batches for DAM and IADAM
       for DAM, conf['intent_dict'] == None
       for IADAM, conf['intent_dict'] != None
       In addition to (c,r,y) for each instance, we also look up the corresponding
       predicted intent vectors for (c,r) from the intent_dict in O(1)
    '''
    _turns_batches = []
    _tt_turns_len_batches = []
    _every_turn_len_batches = []
    _turns_intent_batches = []

    _response_batches = []
    _response_len_batches = []
    _response_intent_batches = []

    _label_batches = []

    batch_len = len(data['y'])/conf['batch_size']

    for batch_index in range(batch_len):
        _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label, _turns_intent, _response_intent = build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        _turns_batches.append(_turns)
        _tt_turns_len_batches.append(_tt_turns_len)
        _every_turn_len_batches.append(_every_turn_len)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        if conf['model_name'] != 'dam':
            _turns_intent_batches.append(_turns_intent)
            _response_intent_batches.append(_response_intent)

        _label_batches.append(_label)

    ans = {
        "turns": _turns_batches, "tt_turns_len": _tt_turns_len_batches, "every_turn_len":_every_turn_len_batches,
        "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches,
        "turns_intent" : _turns_intent_batches, "response_intent": _response_intent_batches
    }

    return ans

# def build_batches_iadam(data, conf, intent_dict, turn_cut_type='tail',
#                         term_cut_type='tail'):
#     '''Build batches for intent-aware DAM model
#     In addition to (c,r,y) for each instance, we also look up the corresponding
#     predicted intent vectors for (c,r) from the intent_dict in O(1)
#     '''
#     _turns_batches = []
#     _tt_turns_len_batches = []
#     _every_turn_len_batches = []
#
#     _response_batches = []
#     _response_len_batches = []
#
#     _label_batches = []
#
#     batch_len = len(data['y']) / conf['batch_size']
#     print('number of batches in one epoch', batch_len)
#
#     for batch_index in range(batch_len):
#         # batch_index is to index the batch in the current epoch
#         # if batch_size = 50, and there are 500 instances in data
#         # than batch_len = 10, batch_index = 0,1,...9
#         _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(
#             data, batch_index, conf, turn_cut_type='tail',
#             term_cut_type='tail')
#
#         _turns_batches.append(_turns)
#         _tt_turns_len_batches.append(_tt_turns_len)
#         _every_turn_len_batches.append(_every_turn_len)
#
#         _response_batches.append(_response)
#         _response_len_batches.append(_response_len)
#
#         _label_batches.append(_label)
#
#     ans = {
#         "turns": _turns_batches, "tt_turns_len": _tt_turns_len_batches,
#         "every_turn_len": _every_turn_len_batches,
#         "response": _response_batches, "response_len": _response_len_batches,
#         "label": _label_batches
#     }
#
#     return ans

def read_dict(word2id_file):
    id2word_dict = dict()
    with open(word2id_file) as fin:
        lines = fin.readlines()
        for index in range(0,len(lines)-1, 2):
            id2word_dict[lines[index+1].strip()] = lines[index].strip()
        print('vocab size: ', len(id2word_dict))
    return id2word_dict


# read intent_vectors.txt for intent vectors in DMN_INTENT model
def read_intent(filename):
    intent_dict = {}
    print('read intent vectors...')
    with open(filename) as fin:
        for l in tqdm(fin):
            to = l.strip().split('\t')
            intent_dict[to[0]] = [float(x) for x in
                                  to[1].split()]  # str to float
    return intent_dict

if __name__ == '__main__':
    # conf = {
    #     "batch_size": 256,
    #     "max_turn_num": 10,
    #     "max_turn_len": 50,
    #     "_EOS_": 28270,
    # }
    # train, val, test = pickle.load(open('../../data/data_small.pkl', 'rb'))
    # print('load data success')
    #
    # train_batches = build_batches(train, conf)
    # val_batches = build_batches(val, conf)
    # test_batches = build_batches(test, conf)
    # print('build batches success')
    #
    # pickle.dump([train_batches, val_batches, test_batches], open('../../data/batches_small.pkl', 'wb'))
    # print('dump success')
    word2id_file = '../../data/ubuntu/word2id'
    id2word_dict = read_dict(word2id_file)
    print(dict(id2word_dict.items()[0:5]))

        


    








    
    


