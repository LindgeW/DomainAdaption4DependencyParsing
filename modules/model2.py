from .layer import *
from .dropout import *
import torch
import torch.nn.functional as F


class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbd=0.01):
        ctx.lmbd = lmbd
        return x.reshape_as(x)

    @staticmethod
    # 输入为forward输出的梯度
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.lmbd * grad_input.neg(), None


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                act,
                nn.Linear(hidden_size, out_size))

    def forward(self, x):
        return self.mlp(x)


class BiafModule(nn.Module):
    def __init__(self, args):
        super(BiafModule, self).__init__()
        self.args = args
        in_features = args.d_model 
        self._activation = nn.ReLU()
        # self._activation = nn.LeakyReLU(0.1)
        # self._activation = nn.ELU()

        self.mlp_arc = NonlinearMLP(in_feature=in_features, out_feature=args.arc_size * 2, activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=in_features, out_feature=args.label_size * 2, activation=self._activation)

        self.arc_biaffine = Biaffine(args.arc_size,
                                     1, bias=(True, False))

        self.label_biaffine = Biaffine(args.label_size,
                                       args.rel_size, bias=(True, True))

        #self.domain_cls = nn.Linear(in_features, 2)
        #self.dom_ws = nn.Parameter(torch.zeros(5))

        #self.experts = nn.ModuleList([nn.Linear(in_features, in_features//2, bias=True) for _ in range(5)])
        #self.experts = nn.ModuleList([MLP(in_features, in_features, in_features//2, nn.ReLU()) for _ in range(5)])
        #self.gate = nn.Linear(2*in_features, 5, bias=False)
        #nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, bert_out):
        if self.training:
            bert_out = timestep_dropout(bert_out, self.args.embed_drop)
        
        arc_feat = self.mlp_arc(bert_out)
        lbl_feat = self.mlp_lbl(bert_out)

        arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
        lbl_head, lbl_dep = lbl_feat.chunk(2, dim=-1)

        if self.training:
            arc_head = timestep_dropout(arc_head, self.args.arc_drop)
            arc_dep = timestep_dropout(arc_dep, self.args.arc_drop)

        arc_score = self.arc_biaffine(arc_dep, arc_head)

        if self.training:
            lbl_head = timestep_dropout(lbl_head, self.args.label_drop)
            lbl_dep = timestep_dropout(lbl_dep, self.args.label_drop)
        
        lbl_score = self.label_biaffine(lbl_dep, lbl_head)
        return arc_score, lbl_score


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.biafs = nn.ModuleDict({
            'src': BiafModule(args),
            'tgt': BiafModule(args)
        })


    def forward(self, x, opt='src'):
        return self.biafs[opt](x)


class ParserModel(nn.Module):
    def __init__(self, model, bert=None):
        super(ParserModel, self).__init__()
        self.bert = bert
        self.model = model

    def base_named_params(self):
        '''
        base_params = []
        for param in self.smodel.parameters():
            if param.requires_grad:
                base_params.append(param)
        return base_params
        '''
        return self.model.named_parameters()

    def bert_named_params(self):
        #bert_params = []
        #for name, param in self.sbert.named_parameters():
            #if param.requires_grad:
        #    bert_params.append((name, param))
        #return bert_params
        return self.bert.named_parameters()

    def forward(self, src_inp, tgt_inp=None, alpha=0.01):
        src_mask = src_inp[1].gt(0)
        src_bert_embed = self.bert(*src_inp)
        arc_score, lbl_score = self.model(src_bert_embed, 'src')

        if self.training and tgt_inp is not None:
            tgt_bert_embed = self.bert(*tgt_inp)
            src_arc, src_lbl = self.model(tgt_bert_embed, 'src')
            tgt_arc, tgt_lbl = self.model(tgt_bert_embed, 'tgt')

            #tgt_entropy = (-1. * F.softmax(tgt_arc, dim=-1) * F.log_softmax(tgt_arc, dim=-1)).sum() / (tgt_arc.size(0) * tgt_arc.size(1)) + (-1. * F.softmax(tgt_lbl, dim=-1) * F.log_softmax(tgt_lbl, dim=-1)).sum() / (tgt_lbl.size(0) * tgt_lbl.size(1) ** 2)

            #bs = min(src_bert_embed.size(0), tgt_bert_embed.size(0))
            #src_repr = src_bert_embed[:bs][:, 0, :]
            #tgt_repr = tgt_bert_embed[:bs][:, 0, :]
            #dist_loss = get_coral_loss(src_repr, tgt_repr)
            dist_loss = F.mse_loss(src_arc.softmax(dim=-1), tgt_arc.softmax(dim=-1)) + F.mse_loss(src_lbl.softmax(dim=-1), tgt_lbl.softmax(dim=-1))
            #dist_loss = torch.mean((src_arc.softmax(dim=-1) - tgt_arc.softmax(dim=-1)).abs()) + torch.mean((src_lbl.softmax(dim=-1) - tgt_lbl.softmax(dim=-1)).abs())
            return arc_score, lbl_score, dist_loss
            
        arc_score2, lbl_score2 = self.model(src_bert_embed, 'tgt')
        arc_score += arc_score2
        lbl_score += lbl_score2
        return arc_score/2, lbl_score/2

