from .layer import *
from .dropout import *
import torch


class BiafModule(nn.Module):
    def __init__(self, args):
        super(BiafModule, self).__init__()
        self.args = args
        in_features = args.d_model 
        self._activation = nn.ReLU()
        # self._activation = nn.LeakyReLU(0.1)
        # self._activation = nn.ELU()
        self.mlp_arc = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.arc_size * 2,
                                    activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.label_size * 2,
                                    activation=self._activation)

        self.arc_biaffine = Biaffine(args.arc_size,
                                     1, bias=(True, False))

        self.label_biaffine = Biaffine(args.label_size,
                                       args.rel_size, bias=(True, True))

    def biaf_ws(self):
        return self.arc_biaffine.weight, self.label_biaffine.weight

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
        self.args = args
        self.bfm = nn.ModuleList([BiafModule(args) for _ in range(3)])

    def base_named_params(self):
        '''
        base_params = []
        for m in self.bfm:
            base_params.extend(list(m.named_parameters()))
        return base_params
        '''
        return self.bfm.named_parameters()
    
    def orth_const(self):
        arc_w1, rel_w1 = self.bfm[0].biaf_ws()
        arc_w2, rel_w2 = self.bfm[1].biaf_ws()
        arc_prod = torch.sum((arc_w1.t() @ arc_w2) ** 2)
        rel_prod = torch.sum((rel_w1.t() @ rel_w2) ** 2)
        dot_prod = arc_prod + rel_prod
        return dot_prod

    def forward(self, x, ty=0):
        return self.bfm[ty](x)


class ParserModel(nn.Module):
    def __init__(self, model, bert):
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
        return self.model.base_named_params()

    def bert_named_params(self):
        #bert_params = []
        #for name, param in self.sbert.named_parameters():
            #if param.requires_grad:
        #    bert_params.append((name, param))
        #return bert_params
        return self.bert.named_parameters()

    def base_fw(self, inp):
        embed = self.bert(*inp)
        arc_outs, lbl_outs = [], []
        for i in range(3):
            arc_score, lbl_score = self.model(embed, i)
            arc_outs.append(arc_score)
            lbl_outs.append(lbl_score)
        return arc_outs, lbl_outs
    
    def orth_loss(self, orth_w=0.01):
        return orth_w * self.model.orth_const()

    def forward(self, inp):
        return self.base_fw(inp)
