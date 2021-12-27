from .layer import *
from .dropout import *


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

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
        return self.model.named_parameters()

    def bert_named_params(self):
        #bert_params = []
        #for name, param in self.sbert.named_parameters():
            #if param.requires_grad:
        #    bert_params.append((name, param))
        #return bert_params
        return self.bert.named_parameters()

    def forward(self, inp):
        embed = self.bert(*inp)
        arc_score, lbl_score = self.model(embed)
        return arc_score, lbl_score, embed

