import models.net as net
import models.iadam_attention as iadam_attention

import bin.train_and_evaluate as train

# configure

# data_small.pkl is the small data for debugging purpose (10K training instances for UDC)
# data.pkl is the whole data  (1M training instances for UDC)

conf = {
    "data_name": "udc",
    "data_path": "../data/udc/data_small.pkl", # data_small.pkl or data.pkl
    "intent_vec_path": "../data/udc/intent_vectors.txt", # path of intent vectors
    "intent_size": 12,  # dimensions of different intent
    "intent_attention_type": "bilinear",  # 'dot', 'bilinear', 'outprod'
    "intent_ffn_od0": 64,  # in iadam-concat ffn 144->64->16 match 576
    "intent_ffn_od1": 16,  # in iadam-concat ffn 144->64->16 match 576
    "intent_loss_weight": 0.2, # in iadam-mtl weight for intent loss; 1-weight for the ranking loss
    "model_name": "iadam-attention", # dam, iadam-concat, iadam-attention, iadam-mtl
    "save_path": "../output/udc/temp/",
    "word_emb_init": None, #"../data/udc/cut_embed_mikolov_200d.pkl", # word_embedding.pkl
    "init_model": None, #should be set for test
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
    "_EOS_": 429498, # 28270, #1 for douban data
    "final_n_class": 1,

    "cnn_3d_oc0": 32,
    "cnn_3d_oc1": 16
}

if conf['model_name'] == 'dam':
    model = net.Net(conf)  # DAM
elif conf['model_name'] == 'iadam-attention':
    model = iadam_attention.Net(conf)  # IADAM-Attention-V4-2/ IART
else:
    raise NameError('model not supported.')

train.train(conf, model)

# test and evaluation, init_model in conf should be set
# test.test(conf, model)
