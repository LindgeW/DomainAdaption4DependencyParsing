from .layer import *
from .dropout import *
import torch
import torch.nn.functional as F
import numpy as np


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


class FFNN(nn.Module):
    def __init__(self, input_dim, latent_dim, act=None, skip=False):
        super(FFNN, self).__init__()
        self.act = nn.Identity() if act is None else act
        self.skip = skip
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        if self.skip:
            h1 += x
        
        h2 = self.fc2(h1)
        if self.skip:
            h2 += h1
        return x, h1, h2


class MINE(nn.Module):
    def __init__(self, input_size, hidden_size, mode='jsd'):
        super(MINE, self).__init__()
        assert mode in ['mine', 'jsd']
        self.mode = mode
        self.layers = nn.Sequential(nn.Linear(2 * input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat((x, x), dim=0)
        idx = torch.randperm(batch_size)
        shuffled_y = y[idx]
        #shuffled_y = torch.cat((y[1:], y[0].unsqueeze(0)), dim=0)
        concat_y = torch.cat((y, shuffled_y), dim=0)
        inputs = torch.cat((tiled_x, concat_y), dim=1)
        logits = self.layers(inputs)
        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        if self.mode == 'mine':
            mi_loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))) # max mine
        else:
            mi_loss = -1 * (-F.softplus(-pred_xy).mean() - F.softplus(pred_x_y).mean())    # max jsd
        
        return mi_loss


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        in_features = args.d_model 

        self._activation = nn.ReLU()
        self.mlp_arc = NonlinearMLP(in_feature=in_features, out_feature=args.arc_size * 2, activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=in_features, out_feature=args.label_size * 2, activation=self._activation)
        self.arc_biaffine = Biaffine(args.arc_size,
                                     1, bias=(True, False))

        self.label_biaffine = Biaffine(args.label_size,
                                       args.rel_size, bias=(True, True))

        self.sh_ffn = FFNN(in_features, in_features, nn.ReLU(), True)
        self.sp_ffn = FFNN(in_features, in_features)
        self.mine = MINE(in_features, in_features, mode='jsd')
        self.proj = nn.Linear(2*in_features, in_features, bias=False)

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

    def freeze_mi(self):
        pass

    def bert_named_params(self):
        #bert_params = []
        #for name, param in self.sbert.named_parameters():
            #if param.requires_grad:
        #    bert_params.append((name, param))
        #return bert_params
        return self.bert.named_parameters()

    def forward(self, src_inp, usp=False):
        if usp:
            bert_embed = self.bert(*src_inp)
            bert_repr = bert_embed[:, 0, :]
            #mask = src_inp[1].gt(0)
            #bert_repr = ((bert_embed * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True))
            sh_outs = self.model.sh_ffn(bert_repr)
            sp_outs = self.model.sp_ffn(bert_repr)
            
            io_loss = self.model.mine(sh_outs[0], sh_outs[2])
            mid_loss = self.model.mine(sh_outs[1], sh_outs[2])
            #ss_loss = -self.mine(sp_outs[2], sh_outs[2])
            ss_loss = torch.sum(torch.matmul(sp_outs[2], sh_outs[2].transpose(-1, -2))**2)

            #io_loss = 1 - F.cosine_similarity(sh_outs[0], sh_outs[2], dim=-1, eps=1e-6).mean()
            #mid_loss = 1 - F.cosine_similarity(sh_outs[1], sh_outs[2], dim=-1, eps=1e-6).mean()
            #ss_loss = F.cosine_similarity(sh_outs[2], sp_outs[2], dim=-1, eps=1e-6).mean()
            print(io_loss.data, mid_loss.data, ss_loss.data)
            return io_loss + 0.5 * mid_loss + 0.01*ss_loss
        else:
            bert_embed = self.bert(*src_inp)
            sh_outs = self.model.sh_ffn(bert_embed)
            sp_outs = self.model.sp_ffn(bert_embed)
            inp = self.model.proj(torch.cat((sh_outs[2], sp_outs[2]), dim=-1))
            arc_score, lbl_score = self.model(inp)
            return arc_score, lbl_score
        
