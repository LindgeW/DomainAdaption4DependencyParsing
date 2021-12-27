import os
import json
from argparse import ArgumentParser


def get_data_path(json_path):
    assert os.path.exists(json_path)
    with open(json_path, 'r', encoding='utf-8') as fr:
        data_opts = json.load(fr)

    print(data_opts)
    return data_opts


def args_config():
    parse = ArgumentParser('Biaffine Parser Argument Configuration')

    parse.add_argument('--cuda', type=int, default=-1, help='training device, default on cpu')

    parse.add_argument('-lr', '--learning_rate', type=float, default=8e-4, help='learning rate of training')
    parse.add_argument('-bt1', '--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parse.add_argument('-bt2', '--beta2', type=float, default=0.98, help='beta2 of Adam optimizer')
    parse.add_argument('-eps', '--eps', type=float, default=1e-8, help='eps of Adam optimizer')
    parse.add_argument('-warmup', '--warmup_step', type=int, default=4000, help='warm up steps for optimizer')
    parse.add_argument('--decay', type=float, default=0.75, help='lr decay rate for optimizer')
    parse.add_argument('--decay_step', type=int, default=10000, help='lr decay steps for optimizer')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    parse.add_argument('--scheduler', choices=['cosine', 'inv_sqrt', 'exponent', 'linear', 'const'], default='const', help='the type of lr scheduler')
    parse.add_argument('--grad_clip', type=float, default=1., help='the max norm of gradient clip')
    
    parse.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate of bert')

    parse.add_argument('--patient', type=int, default=3, help='patient times for early-stopping')
    parse.add_argument('--batch_size', type=int, default=2, help='train batch size')
    parse.add_argument('--test_batch_size', type=int, default=32, help='test batch size')
    parse.add_argument('--epoch', type=int, default=30, help='iteration of training')
    parse.add_argument('--update_steps', type=int, default=1, help='gradient accumulation and update per x steps')

    parse.add_argument('--char_embed_dim', type=int, default=100, help='char embedding size')
    parse.add_argument('--tag_embed_dim', type=int, default=50, help='pos_tag embedding size')

    parse.add_argument('--lstm_depth', type=int, default=3, help='the depth of lstm layer')
    parse.add_argument('--arc_size', type=int, default=640, help='arc mlp size')
    parse.add_argument('--label_size', type=int, default=128, help='label mlp size')

    parse.add_argument('-mpe', '--max_pos_embeddings', default=600, help='max sequence position embeddings')
    parse.add_argument('--use_sine_pos', type=bool, default=True, help='whether use sine & cosine position embeddings')
    parse.add_argument("--src_type", type=str, default='BC', help='sub-layer feature size')
    parse.add_argument("--tgt_type", type=str, default='ZX', help='sub-layer feature size')
    
    parse.add_argument("--d_model", type=int, default=768, help='sub-layer feature size')
    parse.add_argument("--d_ff", type=int, default=1000, help='pwffn inner-layer feature size')
    parse.add_argument("--nb_heads", type=int, default=8, help='sub-layer feature size')
    parse.add_argument("--bert_layer", type=int, default=4, help='the number of encoder layer')

    parse.add_argument('--embed_drop', type=float, default=0.33, help='embedding dropout')
    parse.add_argument('--enc_drop', type=float, default=0.2, help='encoder dropout')
    parse.add_argument('--att_drop', type=float, default=0.1, help='attention dropout')
    parse.add_argument('--arc_drop', type=float, default=0.33, help='Arc MLP dropout')
    parse.add_argument('--label_drop', type=float, default=0.33, help='Label MLP dropout')

    args = parse.parse_args()

    print(vars(args))

    return args
