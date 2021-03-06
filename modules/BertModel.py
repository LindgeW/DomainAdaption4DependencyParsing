from transformers import BertModel
import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from modules.scale_mix import ScalarMix


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class AttnMerge(nn.Module):
    def __init__(self, hn_size):
        super(AttnMerge, self).__init__()
        self.fc = nn.Linear(hn_size, hn_size, bias=False)
        #self.fc = nn.Sequential(
        #        nn.Linear(hn_size, hn_size),
        #        nn.ReLU(),
        #        nn.Linear(hn_size, 1, bias=False))
    
    def forward(self, x):
        # (b, l, d)
        hx = self.fc(x)
        #hx = self.fc(x).tanh()
        alpha = F.softmax(hx, dim=1)
        out = torch.sum(alpha * x, dim=1)
        # (b, d)
        return out


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=1, merge='none', fine_tune=True, use_proj=False, proj_dim=256):
        super(BertEmbedding, self).__init__()
        assert merge in ['none', 'linear', 'mean', 'attn']
        self.merge = merge
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.fine_tune = fine_tune
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)

        self.bert_layers = self.bert.config.num_hidden_layers + 1  # including embedding layer
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size

        if self.merge == 'linear':
            self.scale = ScalarMix(self.nb_layers)
            # self.weighing_params = nn.Parameter(torch.ones(self.num_layers), requires_grad=True)
        elif self.merge == 'attn':
            self.scale = AttnMerge(self.hidden_size)

        if not self.fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        if self.use_proj:
            self.proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
            self.hidden_size = self.proj_dim
        else:
            self.proj = None

    def freeze_layers(self, n):
        assert n >=1 and n <= 12
        freeze_layers = ['embeddings'] + [f'layer.{i}' for i in range(n)]
        for n, p in self.bert.named_parameters():
            p.requires_grad = True
            for ele in freeze_layers:
                if ele in n:
                    p.requires_grad = False

    def save_bert(self, save_dir):
        # saved into config file and model
        assert os.path.isdir(save_dir)
        self.bert.save_pretrained(save_dir)
        print('BERT Saved !!!')

    def forward(self, bert_ids, bert_lens, bert_mask):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param segments: (bz, bpe_seq_len)  ????????????????????????0
        :param bert_mask: (bz, bep_seq_len)  ??????bpe??????
        :param bert_lens: (bz, seq_len)  ??????token??????bpe??????????????????
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        bert_mask = bert_mask.type_as(mask)
        if self.fine_tune:
            #last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)
            last_enc_out, _, all_enc_outs = self.bert(bert_ids, attention_mask=bert_mask, return_dict=False)
        else:
            with torch.no_grad():
                #last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)
                last_enc_out, _, all_enc_outs = self.bert(bert_ids, attention_mask=bert_mask, return_dict=False)

        if self.merge == 'linear':
            enc_out = self.scale(all_enc_outs[-self.nb_layers:])  # (bz, seq_len, 768)

            # encoded_repr = 0
            # soft_weight = F.softmax(self.weighing_params, dim=0)
            # for i in range(self.nb_layers):
            #     encoded_repr += soft_weight[i] * all_enc_outs[i]
            # enc_out = encoded_repr
        elif self.merge == 'mean':
            top_enc_outs = all_enc_outs[-self.nb_layers:]
            enc_out = sum(top_enc_outs) / len(top_enc_outs)
            # enc_out = torch.stack(tuple(top_enc_outs), dim=0).mean(0)
        elif self.merge == 'attn':
            top_enc_out = torch.stack(all_enc_outs[-self.nb_layers:], dim=1)
            enc_out = self.scale(top_enc_out)
        else:
            enc_out = last_enc_out

        bert_chunks = enc_out[bert_mask].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(bz, seq_len, self.bert.config.hidden_size)
        output = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        
        if self.proj:
            return self.proj(output)
        else:
            return output
