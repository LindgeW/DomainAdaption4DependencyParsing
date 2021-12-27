import random
import numpy as np
from conf.config import get_data_path, args_config
from datautil.dataloader import load_dataset
from vocab.dep_vocab import create_vocab
from modules.model import *
# from modules.model2 import *
from modules.parser import BiaffineParser
from modules.BertModel import BertEmbedding


if __name__ == '__main__':
    random.seed(3046)
    np.random.seed(3046)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1344)
    torch.cuda.manual_seed_all(1344)
    torch.backends.cudnn.deterministic = True

    print('cuda available:', torch.cuda.is_available())
    print('cuDnn available:', torch.backends.cudnn.enabled)
    print('GPU numbers:', torch.cuda.device_count())

    data_path = get_data_path("./conf/datapath.json")
    args = args_config()

    dep_vocab = create_vocab(data_path[args.src_type]['train_data'], data_path['pretrained']['bert_model'])
    args.tag_size = dep_vocab.tag_size
    args.rel_size = dep_vocab.rel_size
    ''' 
    # multi-source
    src_type = ['BC', 'ZX', 'PB', 'PC']
    src_train_data = []
    for st in src_type:
        if st != args.tgt_type:
            src_train_data += load_dataset(data_path[st]['train_data'], dep_vocab)
    '''
    src_train_data = load_dataset(data_path[args.src_type]['train_data'], dep_vocab)
    print('src train data size:', len(src_train_data))
    tgt_train_data = load_dataset(data_path[args.tgt_type]['train_data'], dep_vocab)
    print('tgt train data size:', len(tgt_train_data))
    dev_data = load_dataset(data_path[args.tgt_type]['dev_data'], dep_vocab)
    print('dev data size:', len(dev_data))
    test_data = load_dataset(data_path[args.tgt_type]['test_data'], dep_vocab)
    print('test data size:', len(test_data))

    bert_embed = BertEmbedding(data_path['pretrained']['bert_model'], nb_layers=args.bert_layer, merge='linear', use_proj=False, proj_dim=args.d_model)
    base_model = BaseModel(args)
    parser_model = ParserModel(base_model, bert_embed)
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    parser_model = parser_model.to(args.device)
    biff_parser = BiaffineParser(parser_model)
    biff_parser.summary()
    print('模型参数量：', sum(p.numel() for p in parser_model.parameters() if p.requires_grad))

    biff_parser.train(src_train_data, tgt_train_data, dev_data, test_data, args, dep_vocab)

