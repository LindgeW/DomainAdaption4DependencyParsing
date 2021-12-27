from datautil.dataloader import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from modules.decode_alg.eisner import eisner
from log.logger import logger
from .optimizer import *
from datautil.dependency import Dependency
from modules.dep_eval import evaluation
import numpy as np
import copy


def cse_loss(stu_logits, tea_logits, mask=None):
    stu_probs = stu_logits.log_softmax(dim=-1)
    tea_probs = tea_logits.softmax(dim=-1)
    ce = - torch.mul(tea_probs, stu_probs).sum(dim=-1).mean()
    return ce


def coral(source, target):
    d = source.data.size(1)
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t().contiguous() @ xm
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t().contiguous() @ xmt
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss


class Optimizer(object):
    def __init__(self, params, args):
        self.args = args
        self.train_step = 0
        # self.optimizer = optim.Adam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
        self.optimizer = optim.AdamW(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

        lr_scheduler = None
        if args.scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_step, eta_min=1e-6)
        elif args.scheduler == 'exponent':
            def lr_lambda(step):
                return args.decay ** (step / args.decay_step)

            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif args.scheduler == 'inv_sqrt':
            def lr_lambda(step):
                if step == 0 and args.warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)

            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif args.scheduler == 'linear':
            lr_scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=args.warmup_step, t_total=args.max_step)
        else:
            pass

        self.lr_scheduler = lr_scheduler

    def step(self):
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.train_step += 1
            if self.args.scheduler in ['cosine', 'exponent']:
                if self.train_step < self.args.warmup_step:
                    curr_lr = self.args.learning_rate * self.train_step / self.args.warmup_step
                    self.optimizer.param_groups[0]['lr'] = curr_lr
                else:
                    self.lr_scheduler.step(self.train_step)
            else:
                self.lr_scheduler.step(self.train_step)

        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.lr_scheduler.step()

    def get_lr(self):
        # current_lr = self.lr_scheduler.get_lr()[0]
        current_lr = self.optimizer.param_groups[0]['lr']
        return current_lr


# class ScheduleOptimizer(object):
#     def __init__(self, optimizer, d_model, warmup_steps):
#         self.optimizer = optimizer
#         self.warmup_steps = warmup_steps
#         self.init_lr = d_model ** -0.5
#         self.step_num = 0
#
#     def _adjust_lr(self):
#         self.step_num += 1
#         lr = self.init_lr * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
#         for group in self.optimizer.param_groups:
#             group['lr'] = lr
#
#     def step(self):
#         self._adjust_lr()
#         self.optimizer.step()
#
#     def zero_grad(self):
#         self.optimizer.zero_grad()
#
#     def get_lr(self):
#         current_lr = self.optimizer.param_groups[0]['lr']
#         return current_lr


class BiaffineParser(object):
    def __init__(self, model, args):
        super(BiaffineParser, self).__init__()
        self.model = model
        self.orig_model_dict = copy.deepcopy(self.model.state_dict())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
                        'weight_decay': 0.01, 'lr': args.bert_lr},
            {'params': [p for n, p in self.model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
                        'weight_decay': 0.0, 'lr': args.bert_lr},
            
            {'params': [p for n, p in self.model.base_named_params() if p.requires_grad],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate}
        ]
        self.optimizer_bert = optim.AdamW(optimizer_bert_parameters, lr=args.bert_lr, betas=(0.9, 0.99), eps=1e-8)
    
    def summary(self):
        logger.info(self.model)


    def train(self, task_id, src_train_data, dev_data, test_data, args, vocab):
        #optimizer = Optimizer(self.parser_model.base_params(), args)
        '''
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
                        'weight_decay': 0.01, 'lr': args.bert_lr},
            {'params': [p for n, p in self.model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
                        'weight_decay': 0.0, 'lr': args.bert_lr},
            
            {'params': [p for n, p in self.model.base_named_params() if p.requires_grad],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate}
        ]
        self.optimizer_bert = optim.AdamW(optimizer_bert_parameters, lr=args.bert_lr, betas=(0.9, 0.99), eps=1e-8)
        '''

        patient = 0
        dev_best_uas, dev_best_las = 0, 0
        test_best_uas, test_best_las = 0, 0
        #best_model_dict = None
        self.model.load_state_dict(self.orig_model_dict)
        for ep in range(1, 1+args.epoch):
            self.model.train()
            train_loss = 0
            start_time = time.time()
            for i, batch_data in enumerate(batch_iter(src_train_data, args.batch_size, shuffle=True)):
                src_batcher = batch_variable(batch_data, vocab, args.device)
                (bert_ids, bert_lens, bert_mask), true_heads, true_rels = src_batcher
            
                arc_outs, rel_outs = self.model(src_batcher[0])
                if task_id in list(range(3)):
                    loss = self.calc_dep_loss(arc_outs[task_id], rel_outs[task_id], true_heads, true_rels, bert_lens.gt(0))
                else:
                    loss = 0.
                    for arc_score, rel_score in zip(arc_outs, rel_outs):
                        loss += self.calc_dep_loss(arc_score, rel_score, true_heads, true_rels, bert_lens.gt(0))
                    #loss += self.model.orth_loss(0.01)

                loss.backward()  # 反向传播，计算当前梯度
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=args.grad_clip)
                self.optimizer_bert.step()
                self.optimizer_bert.zero_grad()  # 清空过往梯度
                
                loss_val = loss.data.item()
                train_loss += loss_val
                
                logger.info('time cost: %.2fs, train loss: %f' % ((time.time() - start_time), loss_val))

            #log.info('[Epoch %d] train loss: %f, ARC: %.2f%%, REL: %.2f%%' % (ep, train_loss, arc, rel))
            logger.info('[Epoch %d] train loss: %f' % (ep, train_loss))
            
            dev_uas, dev_las = self.evaluate(task_id, dev_data, args, vocab)
            logger.info('Dev data -- UAS: %.2f%%, LAS: %.2f%%' % (dev_best_uas, dev_best_las))
            if dev_best_uas < dev_uas or dev_best_las < dev_las: 
                dev_best_uas = dev_uas
                dev_best_las = dev_las
                test_uas, test_las = self.evaluate(task_id, test_data, args, vocab)
                if test_best_uas < test_uas:
                    test_best_uas = test_uas
                if test_best_las < test_las:
                    test_best_las = test_las
                    #best_model_dict = self.model.state_dict()

                patient = 0
                logger.info('Test data -- UAS: %.2f%%, LAS: %.2f%%' % (test_uas, test_las))
            else:
                patient += 1

            if patient >= args.patient:
                break

        test_uas, test_las = self.evaluate(task_id, test_data, args, vocab)
        if test_best_uas < test_uas:
            test_best_uas = test_uas
        if test_best_las < test_las:
            test_best_las = test_las
            #best_model_dict = self.model.state_dict()
       
        #self.model.load_state_dict(best_model_dict)
        logger.info('Test performance -- UAS: %.2f%%, LAS: %.2f%%' % (test_best_uas, test_best_las))
        return test_best_uas, test_best_las


    def tri_predict(self, test_data, args, vocab):
        self.model.eval()
        pred_heads = {x: [] for x in range(3)}
        pred_rels = {x: [] for x in range(3)}
        with torch.no_grad():
            for batch_data in batch_iter(test_data, args.test_batch_size):
                batcher = batch_variable(batch_data, vocab, args.device)
                bert_lens = batcher[0][1]
                # (b, l, l)   (b, l, l, c)
                arc_outs, rel_outs = self.model(batcher[0])
                # (b, l)  (b, l)
                for i, (arc_score, rel_score) in enumerate(zip(arc_outs, rel_outs)):
                    heads, rels = self.decode(arc_score, rel_score, bert_lens.gt(0))
                    #heads, rels = self.greedy_decode(arc_score, rel_score, bert_lens.gt(0))
                    pred_heads[i].extend(heads.data.tolist())
                    pred_rels[i].extend(rels.data.tolist())
        
        return pred_heads, pred_rels


    def self_predict(self, test_data, args, vocab):
        self.model.eval()
        arc_confs = []
        pred_arcs, pred_rels = [], []
        with torch.no_grad():
            for batch_data in batch_iter(test_data, args.test_batch_size):
                batcher = batch_variable(batch_data, vocab, args.device)
                bert_lens = batcher[0][1]
                arc_outs, rel_outs = self.model(batcher[0])
                mask = bert_lens.gt(0)
                # decoding
                arc_score = arc_outs[0].data
                rel_score = rel_outs[0].data
                heads, rels = self.decode(arc_score, rel_score, mask)
                pred_arcs.extend(heads.data.tolist())
                pred_rels.extend(rels.data.tolist())
                # calculate confidence
                arc_score = arc_score + torch.diag(arc_score.new(arc_score.size(1)).fill_(-1e10))
                arc_score.masked_fill_(~mask.unsqueeze(1), -1e10)
                head_prob = torch.softmax(arc_score, dim=-1).max(dim=-1)[0]
                head_conf = torch.prod(head_prob, dim=-1)
                arc_confs.extend(head_conf.tolist())
        
        #top_ids = np.argsort(arc_conf)[-len(test_data)//2:]
        #print(arc_conf[:10], top_ids[:10])
        #high_conf_data = [x for i, x in enumerate(test_data) if i in top_ids]
        #return high_conf_data
        assert len(arc_confs) == len(pred_arcs) == len(pred_rels)
        return pred_arcs, pred_rels, arc_confs


    def self_train(self, src_train_data, tgt_train_data, dev_data, test_data, args, vocab):
        best_las = 0
        pseudo_data = []
        for ep in range(1, 1+args.epoch):
            base_uas_las = self.train(0, src_train_data+pseudo_data, dev_data, test_data, args, vocab)
            print('test performance:', base_uas_las)
            if base_uas_las[1] > best_las:
                best_las = base_uas_las[1]
            else:
                break
            
            pseudo_data = []
            pred_heads, pred_rels, arc_confs = self.self_predict(tgt_train_data, args, vocab)
            top_ids = np.argsort(arc_confs)[-len(tgt_train_data)//2:]
            for i in top_ids:
                tree = tgt_train_data[i]
                for dep, hid, rel_id in zip(tree, pred_heads[i], pred_rels[i]):
                    dep.head = hid 
                    dep.dep_rel = rel_id
                pseudo_data.append(copy.deepcopy(tree))

        test_uas_las = self.evaluate(0, test_data, args, vocab)
        logger.info('Final test performance -- UAS: %.2f%%, LAS: %.2f%%' % (test_uas_las[0], test_uas_las[1]))
   

    def tri_train(self, src_train_data, tgt_train_data, dev_data, test_data, args, vocab):
        # vanilla traing for encoder and three clser
        base_uas_las = self.train(-1, src_train_data, dev_data, test_data, args, vocab)
        print('base parsing performance:', base_uas_las)

        improved = [True] * 3
        best_las = [base_uas_las[1]] * 3
        for ep in range(1, 1+args.epoch):
            #pseudo_data = [[]] * 3
            pseudo_data = {x: [] for x in range(3)}
            # relabel and choose the unlabeled target-domain training data
            for i in range(3):
                j, k = [x for x in range(3) if x != i]
                pred_heads, pred_rels = self.tri_predict(tgt_train_data, args, vocab)
                pred_heads1, pred_rels1 = pred_heads[j], pred_rels[j]
                pred_heads2, pred_rels2 = pred_heads[k], pred_rels[k]

                for x, tgt_inst in enumerate(tgt_train_data):
                    if pred_rels1[x] == pred_rels2[x]:
                    #if pred_heads1[x] == pred_heads2[x] and pred_rels1[x] == pred_rels2[x]:
                        for dep, hid, rel_id in zip(tgt_inst, pred_heads1[x], pred_rels1[x]):
                            dep.head = hid 
                            dep.dep_rel = rel_id
                    else:
                        continue
                    pseudo_data[i].append(copy.deepcopy(tgt_inst))

            # retrain model on pseudo data
            for j in range(3):
                if improved[j] and len(pseudo_data[j]) > 0:
                    print(j, len(pseudo_data[j]))
                    if j == 2:
                        update_uas_las = self.train(j, pseudo_data[j], dev_data, test_data, args, vocab)
                    else:
                        update_uas_las = self.train(j, src_train_data+pseudo_data[j], dev_data, test_data, args, vocab)
                    if update_uas_las[1] > best_las[j]:
                        best_las[j] = update_uas_las[1]
                    else:
                        improved[j] = False
                else:
                    improved[j] = False
            
            if improved == [False] * 3:
                break

        test_uas_las = self.evaluate(-1, test_data, args, vocab)
        logger.info('Final test performance -- UAS: %.2f%%, LAS: %.2f%%' % (test_uas_las[0], test_uas_las[1]))


    def evaluate(self, task_id, test_data, args, vocab):
        self.model.eval()
        all_gold_heads, all_gold_rels = [], []
        all_pred_heads, all_pred_rels = [], []
        with torch.no_grad():
            for batch_data in batch_iter(test_data, args.test_batch_size):
                batcher = batch_variable(batch_data, vocab, args.device)
                (bert_ids, bert_lens, bert_mask), true_heads, true_rels = batcher
                
                arc_outs, rel_outs = self.model(batcher[0])
                if task_id in list(range(3)):
                    arc_score, rel_score = arc_outs[task_id], rel_outs[task_id]
                else:
                    arc_score, rel_score = sum(arc_outs)/len(arc_outs), sum(rel_outs)/len(rel_outs)   
                    '''
                    # major voting for test data
                    pred_heads, pred_rels = {x: [] for x in range(3)}, {x: [] for x in range(3)}
                    for i, (arc_score, rel_score) in enumerate(zip(arc_outs, rel_outs)):
                        heads, rels = self.decode(arc_score, rel_score, bert_lens.gt(0))
                        #heads, rels = self.greedy_decode(arc_score, rel_score, bert_lens.gt(0))
                        pred_heads[i].extend(heads.data.tolist())
                        pred_rels[i].extend(rels.data.tolist())
                    pred_heads = np.asarray(list(pred_heads.values()))
                    pred_rels = np.asarray(list(pred_rels.values()))
                    pred_heads[0][pred_heads[1] == pred_heads[2]] = pred_heads[1][pred_heads[1] == pred_heads[2]] 
                    pred_rels[0][pred_rels[1] == pred_rels[2]] = pred_rels[1][pred_rels[1] == pred_rels[2]] 
                    pred_heads, pred_rels = pred_heads[0], pred_rels[0]
                    '''
                
                pred_heads, pred_rels = self.decode(arc_score, rel_score, bert_lens.gt(0))
                for i, sent in enumerate(batch_data):
                    l = len(sent)
                    all_gold_heads.append(true_heads[i][1:l].tolist())
                    all_gold_rels.append(true_rels[i][1:l].tolist())
                    all_pred_heads.append(pred_heads[i][1:l].tolist())
                    all_pred_rels.append(pred_rels[i][1:l].tolist())
                
        uas, las = evaluation(all_gold_heads, all_gold_rels, all_pred_heads, all_pred_rels)
        return uas, las


    def greedy_decode(self, pred_arc_score, pred_rel_score, mask):
        '''
        :param pred_arc_score: (bz, seq_len, seq_len)
        :param pred_rel_score: (bz, seq_len, seq_len, rel_size)
        :param mask: (bz, seq_len)  pad部分为0
        :return: pred_heads (bz, seq_len)
                 pred_rels (bz, seq_len)
        '''
        bz, seq_len, _ = pred_arc_score.size()
        matrix = pred_arc_score + torch.diag(pred_arc_score.new(seq_len).fill_(-1e10))
        matrix.masked_fill_(~mask.unsqueeze(1), -1e10)
        pred_heads = matrix.data.argmax(dim=2)
        if mask is not None:
            pred_heads *= mask.long()
        #pred_heads = pred_arc_score.data.argmax(dim=-1)
        pred_rels = pred_rel_score.data.argmax(dim=-1)
        # pred_rels = pred_rels.gather(dim=-1, index=pred_heads.unsqueeze(-1)).squeeze(-1)
        pred_rels = pred_rels[torch.arange(bz, dtype=torch.long, device=pred_arc_score.device).unsqueeze(1),
                              torch.arange(seq_len, dtype=torch.long, device=pred_arc_score.device).unsqueeze(0),
                              pred_heads].contiguous()
        return pred_heads, pred_rels

        
    def decode(self, pred_arc_score, pred_rel_score, mask):
        '''
        :param pred_arc_score: (bz, seq_len, seq_len)
        :param pred_rel_score: (bz, seq_len, seq_len, rel_size)
        :param mask: (bz, seq_len)  pad部分为0
        :return: pred_heads (bz, seq_len)
                 pred_rels (bz, seq_len)
        '''
        bz, seq_len, _ = pred_arc_score.size()
        # pred_heads = mst_decode(pred_arc_score, mask)
        mask[:, 0] = 0  # mask out <root>
        pred_heads = eisner(pred_arc_score, mask)
        pred_rels = pred_rel_score.data.argmax(dim=-1)
        # pred_rels = pred_rels.gather(dim=-1, index=pred_heads.unsqueeze(-1)).squeeze(-1)
        pred_rels = pred_rels[torch.arange(bz, dtype=torch.long, device=pred_arc_score.device).unsqueeze(1),
                              torch.arange(seq_len, dtype=torch.long, device=pred_arc_score.device).unsqueeze(0), pred_heads].contiguous()
        return pred_heads, pred_rels


    def calc_dep_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
        '''
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 有效部分mask
        :return:
        '''
        # non_pad_mask[:, 0] = 0  # mask out <root>
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        masked_true_heads = true_heads.masked_fill(pad_mask, -1)
        arc_loss = F.cross_entropy(pred_arcs.transpose(1, 2), masked_true_heads, ignore_index=-1)

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()

        masked_true_rels = true_rels.masked_fill(pad_mask, -1)
        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = F.cross_entropy(out_rels.transpose(1, 2), masked_true_rels, ignore_index=-1)
        return arc_loss + rel_loss


    def calc_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
        '''a
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 非填充部分mask
        :return:
        '''
        # non_pad_mask[:, 0] = 0  # mask out <root>
        bz, seq_len, seq_len, rel_size = pred_rels.size()

        # (bz, seq_len)
        pred_heads = pred_arcs.data.argmax(dim=2)
        arc_acc = ((pred_heads == true_heads) * non_pad_mask).sum().item()

        total_arcs = non_pad_mask.sum().item()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()
        pred_rels = out_rels.data.argmax(dim=2)
        rel_acc = ((pred_rels == true_rels) * non_pad_mask).sum().item()

        return arc_acc, rel_acc, total_arcs

    # def calc_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
    #     '''
    #     :param pred_arcs: (bz, seq_len, seq_len)
    #     :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    #     :param true_heads: (bz, seq_len)  包含padding
    #     :param true_rels: (bz, seq_len)
    #     :param non_pad_mask: (bz, seq_len) 有效部分mask
    #     :return:
    #     '''
    #     non_pad_mask = non_pad_mask.byte()
    #     non_pad_mask[:, 0] = 0
    #
    #     pred_heads = pred_arcs[non_pad_mask]  # (bz, seq_len)
    #     true_heads = true_heads[non_pad_mask]   # (bz, )
    #     pred_rels = pred_rels[non_pad_mask]  # (bz, seq_len, rel_size)
    #     pred_rels = pred_rels[torch.arange(len(pred_rels), dtype=torch.long), true_heads]  # (bz, rel_size)
    #     true_rels = true_rels[non_pad_mask]     # (bz, )
    #
    #     arc_loss = F.cross_entropy(pred_heads, true_heads)
    #     rel_loss = F.cross_entropy(pred_rels, true_rels)
    #
    #     return arc_loss + rel_loss

    # def calc_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
    #     '''
    #     :param pred_arcs: (bz, seq_len, seq_len)
    #     :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    #     :param true_heads: (bz, seq_len)  包含padding
    #     :param true_rels: (bz, seq_len)
    #     :param non_pad_mask: (bz, seq_len)
    #     :return:
    #     '''
    #     non_pad_mask = non_pad_mask.byte()
    #
    #     pred_heads = pred_arcs[non_pad_mask]  # (bz, seq_len)
    #     true_heads = true_heads[non_pad_mask]  # (bz, )
    #     pred_heads = pred_heads.data.argmax(dim=-1)
    #     arc_acc = true_heads.eq(pred_heads).sum().item()
    #     total_arcs = non_pad_mask.sum().item()
    #
    #     pred_rels = pred_rels[non_pad_mask]  # (bz, seq_len, rel_size)
    #     pred_rels = pred_rels[torch.arange(len(pred_rels)), true_heads]  # (bz, rel_size)
    #     pred_rels = pred_rels.data.argmax(dim=-1)
    #     true_rels = true_rels[non_pad_mask]  # (bz, )
    #     rel_acc = true_rels.eq(pred_rels).sum().item()
    #
    #     return arc_acc, rel_acc, total_arcs
