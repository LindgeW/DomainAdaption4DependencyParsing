from .layer import *
from .dropout import *
import torch
import torch.nn.functional as F
from .mmd import *


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


class CoVWeightingLoss(nn.Module):
    def __init__(self, num_losses=5, mean_decay_param=1., mean_sort='full'):
        super(CoVWeightingLoss, self).__init__()
        self.num_losses = num_losses
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if mean_sort == 'decay' else False
        self.mean_decay_param = mean_decay_param  # default 1
        self.current_iter = -1
        
        alphas = torch.zeros((self.num_losses,), requires_grad=False)
        running_mean_L = torch.zeros((self.num_losses,), requires_grad=False)
        running_mean_l = torch.zeros((self.num_losses,), requires_grad=False)
        running_S_l = torch.zeros((self.num_losses,), requires_grad=False)
        self.register_buffer('alphas', alphas)
        self.register_buffer('running_mean_L', running_mean_L)
        self.register_buffer('running_mean_l', running_mean_l)
        self.register_buffer('running_S_l', running_S_l)
        self.running_std_l = None

    def forward(self, unweighted_losses: list):
        device = unweighted_losses[0].device
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(device)

        if not self.training:
            return torch.sum(L)

        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,)).to(device) / self.num_losses
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L
        print(self.alphas)
        # Get the weighted losses and perform a standard back-pass.
        loss = sum([self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))])
        return loss


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden2mean = nn.Linear(input_dim, latent_dim)
        self.hidden2logv = nn.Linear(input_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, input_dim)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.hidden2mean.weight)
        nn.init.xavier_uniform_(self.hidden2logv.weight)
        nn.init.xavier_uniform_(self.latent2hidden.weight)

    def forward(self, h):
        mean = self.hidden2mean(h)
        logv = self.hidden2logv(h)
        std = torch.exp(0.5 * logv)
        z = torch.randn((h.size(0), self.latent_dim), device=h.device)
        z = z * std + mean
        restruct_hidden = self.latent2hidden(z)
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp()) / logv.size(0)
        dist_loss = F.mse_loss(restruct_hidden, h)
        return dist_loss, kl_loss


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
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
        #self.src_proj = nn.Linear(in_features, in_features)
        #self.tgt_proj = nn.Linear(in_features, in_features)
        
        #self.log_vars = nn.Parameter(torch.zeros(5, dtype=torch.float32), requires_grad=True)
        #self.cov_ws = CoVWeightingLoss(5)

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
        #src_mask = src_inp[1].gt(0)
        src_bert_embed = self.bert(*src_inp)
        arc_score, lbl_score = self.model(src_bert_embed)
        if tgt_inp is not None:
            #tgt_mask = tgt_inp[1].gt(0)
            tgt_bert_embed = self.bert(*tgt_inp)
            src_bs, tgt_bs = src_bert_embed.size(0), tgt_bert_embed.size(0)
            
            '''
            src_repr = src_bert_embed[:, 0, :]
            tgt_repr = tgt_bert_embed[:, 0, :]
            if src_bs > tgt_bs:
                exp_t = (src_bs + tgt_bs - 1) // tgt_bs
                tgt_repr = tgt_repr.repeat(exp_t, 1)[:src_bs]
            elif tgt_bs > src_bs:
                exp_t = (src_bs + tgt_bs - 1) // src_bs
                src_repr = src_repr.repeat(exp_t, 1)[:tgt_bs]
            '''

            bs = min(src_bs, tgt_bs)
            src_repr = src_bert_embed[:bs][:, 0, :]
            tgt_repr = tgt_bert_embed[:bs][:, 0, :]
            #src_repr = self.model.src_proj(src_bert_embed[:bs][:, 0, :])
            #tgt_repr = self.model.tgt_proj(tgt_bert_embed[:bs][:, 0, :])
            #src_repr = src_bert_embed[:bs].mean(dim=1)
            #tgt_repr = tgt_bert_embed[:bs].mean(dim=1)
            #src_repr = ((src_bert_embed * src_mask.unsqueeze(-1)).sum(dim=1) / src_mask.sum(dim=1, keepdim=True))[:bs]
            #tgt_repr = ((tgt_bert_embed * tgt_mask.unsqueeze(-1)).sum(dim=1) / tgt_mask.sum(dim=1, keepdim=True))[:bs]
           
            '''            
            src_mask_expanded = src_mask.unsqueeze(-1).expand(src_bert_embed.size()).float()
            src_sum_embeddings = torch.sum(src_bert_embed * src_mask_expanded, 1)
            src_sum_mask = torch.clamp(src_mask_expanded.sum(1), min=1e-9)
            src_repr = src_sum_embeddings / src_sum_mask
            tgt_mask_expanded = tgt_mask.unsqueeze(-1).expand(tgt_bert_embed.size()).float()
            tgt_sum_embeddings = torch.sum(tgt_bert_embed * tgt_mask_expanded, 1)
            tgt_sum_mask = torch.clamp(tgt_mask_expanded.sum(1), min=1e-9)
            tgt_repr = tgt_sum_embeddings / tgt_sum_mask
            src_repr = src_repr[:bs]
            tgt_repr = tgt_repr[:bs]
            '''

            ''' 
            src_pred = self.model.domain_cls(GRLayer.apply(src_repr, alpha))
            tgt_pred = self.model.domain_cls(GRLayer.apply(tgt_repr, alpha))
            dom_loss = F.cross_entropy(src_pred, torch.ones(src_pred.size(0), dtype=torch.long, device=src_pred.device)) + F.cross_entropy(tgt_pred, torch.zeros(tgt_pred.size(0), dtype=torch.long, device=tgt_pred.device))
            '''
            
            #dist_loss = get_coral_loss(src_repr, tgt_repr)
            dist_loss = mmd(src_repr, tgt_repr)
            #tgt1 = self.model.src_proj(tgt_bert_embed[:bs].sum(dim=1))
            #tgt2 = self.model.tgt_proj(tgt_bert_embed[:bs].sum(dim=1))
            #dist_loss += torch.mean(torch.abs(F.softmax(tgt1, dim=-1) - F.softmax(tgt2, dim=-1)))
            #dist_loss += torch.sum((F.softmax(tgt1, dim=-1) - F.softmax(tgt2, dim=-1)) ** 2)

            '''
            mse_loss = F.mse_loss(src_repr, tgt_repr)
            cos_loss = (1. - F.cosine_similarity(src_repr, tgt_repr, dim=-1, eps=1e-6)).mean()
            kl_loss = (F.kl_div(tgt_repr.log_softmax(dim=-1), src_repr.softmax(dim=-1), reduction='batchmean') + F.kl_div(src_repr.log_softmax(dim=-1), tgt_repr.softmax(dim=-1), reduction='batchmean')) / 2.
            mmd_loss = mmd(src_repr, tgt_repr)
            coral_loss = get_coral_loss(src_repr, tgt_repr)
            losses = [mse_loss, cos_loss, kl_loss, mmd_loss, coral_loss]
            #dist_loss = sum([torch.exp(-self.model.log_vars[i]) * loss + self.model.log_vars[i] for i, loss in enumerate(losses)])
            dist_loss = self.model.cov_ws(losses)
            '''

            return arc_score, lbl_score, dist_loss

        return arc_score, lbl_score

