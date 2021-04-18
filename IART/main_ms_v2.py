import models.net as net
import models.iadam_attention as iadam_attention

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

# configure

# data_small.pkl is the small data for debugging purpose (10K training instances)
# data.pkl is the whole data

conf = {
    "data_name": "ms_v2",
    "data_path": "../data/ms_v2/data_small.pkl", # data_small.pkl or data.pkl
    "intent_vec_path": "../data/ms_v2/intent_vectors.txt", # path of intent vectors
    "intent_size": 12, # dimensions of different intent
    "intent_attention_type":"bilinear", # in iadam-attention:  'dot', 'bilinear', 'outprod'
    "intent_ffn_od0": 128, # in iadam-concat ffn 144->128->64 match 6400
    "intent_ffn_od1": 64, # in iadam-concat ffn 144->128->64 match 6400
    "intent_loss_weight": 0.2, # in iadam-mtl weight for intent loss; 1-weight for the ranking loss
    "model_name": "iadam-attention", # dam, iadam-concat, iadam-attention, iadam-mtl
    "save_path": "../output/ms_v2/temp/",
    "word_emb_init": None, # "../data/ms_v2/cut_embed_mikolov_200d.pkl", # None (set None during debugging)
    "init_model": None, # "../output/ms_v2/iadam-attention-max_turn_len-200-run33/model.ckpt.20", # "../output/ms_v2/dam_default_setting_0412_run29/model.ckpt.36", # Set None for training; Set best ckpt for test
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
    "batch_size": 32, # for ms_v2/iadam_mtl model, batch_size = 20; others = 32

    "max_turn_num": 6,  #  6 is better for ms_v2
    "max_turn_len": 200, # default is 180

    "max_to_keep": 1,
    "num_scan_data": 5,  # about 18 hours for 5 epoches on ms_v2
    "_EOS_": 167983, #1 for douban data
    "final_n_class": 1,

    "cnn_3d_oc0": 16,
    "cnn_3d_oc1": 16
}

if conf['model_name'] == 'dam':
    model = net.Net(conf) # DAM
elif conf['model_name'] == 'iadam-attention':
    model = iadam_attention.Net(conf) # IADAM is IART in paper
else:
    raise NameError('model not supported.')

train.train(conf, model)

# test and evaluation, init_model in conf should be set
# test.test(conf, model)

