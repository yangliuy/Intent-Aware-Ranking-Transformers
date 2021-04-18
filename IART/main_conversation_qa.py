import sys
import argparse

import models.net as net
import models.iadam_attention as iadam_attention

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

def main(argv):
    # conf_udc and conf_ms are the default settings for udc and ms_v2
    conf_udc = {
        "data_name": "udc",
        "data_path": "../data/udc/data.pkl",  # data_small.pkl or data.pkl or data_nofd.pkl
        "intent_vec_path": "../data/udc/intent_vectors.txt", # path of intent vectors
        "intent_size": 12,  # dimensions of different intent
        "intent_attention_type": "bilinear",  # 'dot', 'bilinear', 'outprod'. default is bilinear
        "intent_ffn_od0": 64,  # in iadam-concat ffn 144->64->16 match 576
        "intent_ffn_od1": 16,  # in iadam-concat ffn 144->64->16 match 576
        "intent_loss_weight": 0.2,
    # in iadam-mtl weight for intent loss; 1-weight for the ranking loss
        "model_name": "iadam-concat", # dam, iadam-concat, iadam-attention, iadam-mtl
        "save_path": "../output/udc/temp/",
        "word_emb_init": "../data/udc/cut_embed_mikolov_200d.pkl", # word_embedding.pkl
        "init_model": None,  # should be set for test
        "rand_seed": None,
        "drop_dense": None,
        "drop_attention": None,

        "is_mask": True,
        "is_layer_norm": True,
        "is_positional": False,

        "stack_num": 5,
        "attention_type": "dot",

        "learning_rate": 1e-3,
        "vocab_size": 429498,
        "emb_size": 200,
        "batch_size": 128, # for udc/iadam_mtl model, batch_size = 64; others = 128

        "max_turn_num": 9,
        "max_turn_len": 50,

        "max_to_keep": 1,
        "num_scan_data": 2,  # about 16 hours for 2 epoches on udc
        "_EOS_": 429498,  # 28270, #1 for douban data
        "final_n_class": 1,

        "cnn_3d_oc0": 32,
        "cnn_3d_oc1": 16
    }

    conf_ms = {
        "data_name": "ms_v2",
        "data_path": "../data/ms_v2/data.pkl",  # data_small.pkl or data.pkl or data_nofd.pkl
        "intent_vec_path": "../data/ms_v2/intent_vectors.txt", # path of intent vectors
        "intent_size": 12,  # dimensions of different intent
        "intent_attention_type": "bilinear",  # 'dot', 'bilinear', 'outprod'. default is bilinear
        "intent_ffn_od0": 128,  # in iadam-concat ffn 144->128->64 match 6400
        "intent_ffn_od1": 64,  # in iadam-concat ffn 144->128->64 match 6400
        "intent_loss_weight": 0.2,
    # in iadam-mtl weight for intent loss; 1-weight for the ranking loss
        "model_name": "iadam-concat", # dam, iadam-concat, iadam-attention, iadam-mtl
        "save_path": "../output/ms_v2/temp/",
        "word_emb_init": "../data/ms_v2/cut_embed_mikolov_200d.pkl", # "../data/ms_v2/cut_embed_mikolov_200d.pkl", # None (set None during debugging)
        "init_model": None, # "../output/ms_v2/dam_default_setting_0412_run29/model.ckpt.36", #should be set for test

        "rand_seed": None,

        "drop_dense": None,
        "drop_attention": None,

        "is_mask": True,
        "is_layer_norm": True,
        "is_positional": False,

        "stack_num": 4,
        "attention_type": "dot",

        "learning_rate": 1e-3,
        "vocab_size": 167983,
        "emb_size": 200,
        "batch_size": 32,  # 200 for test  256

        "max_turn_num": 6, #  6 is better for ms_v2
        "max_turn_len": 180,

        "max_to_keep": 1,
        "num_scan_data": 5,  # about 18 hours for 5 epoches on ms_v2
        "_EOS_": 167983,  # 1 for douban data
        "final_n_class": 1,

        "cnn_3d_oc0": 16,
        "cnn_3d_oc1": 16
    }

    parser = argparse.ArgumentParser()
    # python main_conversation_qa.py --help to print the help messages
    # sys.argv includes a list of elements starting with the program
    # required parameters
    parser.add_argument('--phase', default='train',
                        help='phase: it can be train or predict, the default \
                        value is train.',
                        required=True)
    parser.add_argument('--data_name', default='udc',
                        help='data_name: name of the data. it can be udc or \
                             ms_v2', required=True)
    parser.add_argument('--model_name', default='dam',
                        help='model_name: name of the model', required=True)
    parser.add_argument('--save_path', default='../output/udc/temp/',
                        help='save_path: output path for model files, score \
                             files and result files', required=True)
    parser.add_argument('--or_cmd', default=False,
                        help='or_cmd: whether want to override config \
                        parameters by command line parameters',
                        required=True)

    # optional parameters
    parser.add_argument('--intent_vec_path',
                        help='intent_vec_path: path of intent vectors.')
    parser.add_argument('--intent_attention_type',
                        help='intent_attention_type: type of intent attention.')
    parser.add_argument('--intent_ffn_od0',
                        help='intent_ffn_od0: output dimension 0 in FFN for \
                             intent transformation in IADAM-Concat')
    parser.add_argument('--intent_ffn_od1',
                        help='intent_ffn_od1: output dimension 1 in FFN for \
                                 intent transformation in IADAM-Concat')
    parser.add_argument('--intent_loss_weight',
                        help='intent_loss_weight: weight of intent loss \
                                     in IADAM-MTL model')
    parser.add_argument('--data_path',
                        help='data_path: path of input data.')
    parser.add_argument('--word_emb_init',
                        help='data_name: path of word embedding file to \
                        initialize the word embeddings.')
    parser.add_argument('--init_model',
                        help='init_model: path of the checkpoints of \
                        model initialization during testing phase.')
    parser.add_argument('--rand_seed',
                        help='rand_seed: rand seed used in numpy.')
    parser.add_argument('--is_positional',
                        help='is_positional: whether add positional embeddings.')
    parser.add_argument('--stack_num',
                        help='stack_num: stack number in Transformers.')
    parser.add_argument('--attention_type',
                        help='attention_type: attention_type in attentive module \
                        in Transformers (dot or bilinear).')  # Added in net.py
    parser.add_argument('--learning_rate',
                        help='learning_rate: initial learning rate in \
                        exponential decay learning rate.')
    parser.add_argument('--vocab_size',
                        help='vocab_size: vocabulary size.')
    parser.add_argument('--emb_size',
                        help='emb_size: embedding size.')
    parser.add_argument('--batch_size',
                        help='batch_size: batch size.')
    parser.add_argument('--max_turn_num',
                        help='max_turn_num: max number of turns in conversation \
                        context.')
    parser.add_argument('--max_turn_len',
                        help='max_turn_len: max length of conversation turns.')
    parser.add_argument('--max_to_keep',
                        help='max_to_keep: max number of checkpoints file to \
                        keep.')
    parser.add_argument('--num_scan_data',
                        help='num_scan_data: number of times to scan the data \
                        which is also number of epoches.')
    parser.add_argument('--eos',
                        help='eos: word id for _EOS_, which is the seperator \
                             between different turns in context')
    parser.add_argument('--cnn_3d_oc0',
                        help='cnn_3d_oc0: out_channels_0 of 3D CNN layer.')
    parser.add_argument('--cnn_3d_oc1',
                        help='cnn_3d_oc1: out_channels_1 of 3D CNN layer.')

    args = parser.parse_args()
    # parse the hyper-parameters from the command lines
    phase = args.phase
    or_cmd = bool(args.or_cmd)
    conf = conf_udc if args.data_name == 'udc' else conf_ms
    conf['save_path'] = args.save_path
    conf['model_name'] = args.model_name

    # load settings from the config file
    # then update the hyper-parameters in the config files with the settings
    # passed from command lines
    if or_cmd:
        if args.intent_vec_path != None:
            conf['intent_vec_path'] = args.intent_vec_path
        if args.intent_ffn_od0 != None:
            conf['intent_ffn_od0'] = int(args.intent_ffn_od0)
        if args.intent_ffn_od1 != None:
            conf['intent_ffn_od1'] = int(args.intent_ffn_od1)
        if args.intent_attention_type != None:
            conf['intent_attention_type'] = args.intent_attention_type
        if args.intent_loss_weight != None:
            conf['intent_loss_weight'] = float(args.intent_loss_weight)
        if args.data_path != None:
            conf['data_path'] = args.data_path
        if args.word_emb_init != None:
            conf['word_emb_init'] = args.word_emb_init
        if args.init_model != None:
            conf['init_model'] = args.init_model
        if args.rand_seed != None:
            conf['rand_seed'] = float(args.rand_seed)
        if args.is_positional != None:
            conf['is_positional'] = args.is_positional
        if args.stack_num != None:
            conf['stack_num'] = int(args.stack_num)
        if args.attention_type != None:
            conf['attention_type'] = args.attention_type
        if args.learning_rate != None:
            conf['learning_rate'] = float(args.learning_rate)
        if args.vocab_size != None:
            conf['vocab_size'] = int(args.vocab_size)
        if args.emb_size != None:
            conf['emb_size'] = int(args.emb_size)
        if args.batch_size != None:
            conf['batch_size'] = int(args.batch_size)
        if args.max_turn_num != None:
            conf['max_turn_num'] = int(args.max_turn_num)
        if args.max_turn_len != None:
            conf['max_turn_len'] = int(args.max_turn_len)
        if args.max_to_keep != None:
            conf['max_to_keep'] = int(args.max_to_keep)
        if args.num_scan_data != None:
            conf['num_scan_data'] = int(args.num_scan_data)
        if args.eos != None:
            conf['_EOS_'] = int(args.eos)
        if args.cnn_3d_oc0 != None:
            conf['cnn_3d_oc0'] = int(args.cnn_3d_oc0)
        if args.cnn_3d_oc1 != None:
            conf['cnn_3d_oc1'] = int(args.cnn_3d_oc1)

    if conf['model_name'] == 'dam':
        model = net.Net(conf)  # DAM
    elif conf['model_name'] == 'iadam-attention':
        model = iadam_attention.Net(conf)  # IADAM-Attention-V4-2 (IART)
    else:
        raise NameError('model not supported.')

    if phase == 'train':
        train.train(conf, model)
    elif phase == 'predict':
        # test and evaluation, init_model in conf should be set
        test.test(conf, model)
    else:
        print 'Phase Error.'
    return


if __name__ == '__main__':
    main(sys.argv)

