from datautil.dataloader import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from modules.decode_alg.eisner import eisner
from log.logger import logger
from .optimizer import *
from modules.dep_eval import evaluation


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
    def __init__(self, parser_model):
        super(BiaffineParser, self).__init__()
        assert isinstance(parser_model, nn.Module)
        self.parser_model = parser_model

    def summary(self):
        logger.info(self.parser_model)

    def train(self, src_train_data, tgt_train_data, dev_data, test_data, args, vocab):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.parser_model.bert_named_params() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in self.parser_model.bert_named_params() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
            {'params': [p for n, p in self.parser_model.base_named_params() if p.requires_grad],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate}
        ]
        optimizer_bert = optim.AdamW(optimizer_bert_parameters, lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8)
        
        patient = 0
        pi = 6
        dev_best_uas, dev_best_las = 0, 0
        test_best_uas, test_best_las = 0, 0
        max_step = args.epoch * (len(src_train_data) // args.batch_size)
        for ep in range(1, 1+args.epoch):
            self.parser_model.train()
            train_loss = 0
            all_arc_acc, all_rel_acc, all_arcs = 0, 0, 0
            start_time = time.time()
            if ep <= pi:
                train_data = src_train_data + tgt_train_data 
                #train_data = [src_train_data, tgt_train_data][ep % 2] 
            else: 
                train_data = src_train_data

            for i, batch_data in enumerate(batch_iter(train_data, args.batch_size, shuffle=True)):
                batcher = batch_variable(batch_data, vocab, args.device)
                (bert_ids, bert_lens, bert_mask), true_heads, true_rels = batcher
            
                if ep <= pi:
                    loss = self.parser_model(batcher[0], True)
                else:
                    #self.parser_model.freeze_mi()
                    arc_score, rel_score = self.parser_model(batcher[0], False)
                    loss = self.calc_dep_loss(arc_score, rel_score, true_heads, true_rels, bert_lens.gt(0))

                loss.backward()  # 反向传播，计算当前梯度
                
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parser_model.parameters()), max_norm=args.grad_clip)
                optimizer_bert.step()
                self.parser_model.zero_grad()  # 清空过往梯度

                loss_val = loss.data.item()
                train_loss += loss_val
                if ep > pi:
                    arc_acc, rel_acc, nb_arcs = self.calc_acc(arc_score, rel_score, true_heads, true_rels, bert_lens.gt(0))
                    all_arc_acc += arc_acc
                    all_rel_acc += rel_acc
                    all_arcs += nb_arcs
                    ARC = all_arc_acc * 100. / all_arcs
                    REL = all_rel_acc * 100. / all_arcs
                    logger.info('Iter%d ARC: %.2f%%, REL: %.2f%%' % (i + 1, ARC, REL))
                
                logger.info('Iter%d time cost: %.2fs, train loss: %f' % (i + 1, (time.time() - start_time), loss_val))

            if ep <= pi:
                continue

            arc = all_arc_acc * 100. / all_arcs
            rel = all_rel_acc * 100. / all_arcs
            train_loss /= len(src_train_data)
            logger.info('[Epoch %d] train loss: %f, ARC: %.2f%%, REL: %.2f%%' % (ep, train_loss, arc, rel))
            dev_uas, dev_las = self.evaluate(dev_data, args, vocab)
            if dev_best_uas < dev_uas or dev_best_las < dev_las: 
                dev_best_uas = dev_uas
                dev_best_las = dev_las
                test_uas, test_las = self.evaluate(test_data, args, vocab)
                if test_best_uas < test_uas:
                    test_best_uas = test_uas
                if test_best_las < test_las:
                    test_best_las = test_las

                patient = 0
                logger.info('Dev data -- UAS: %.2f%%, LAS: %.2f%%' % (dev_best_uas, dev_best_las))
                logger.info('Test data -- UAS: %.2f%%, LAS: %.2f%%' % (test_uas, test_las))
            else:
                patient += 1

            if patient >= 3:
                break

        test_uas, test_las = self.evaluate(test_data, args, vocab)
        if test_best_uas < test_uas:
            test_best_uas = test_uas
        if test_best_las < test_las:
            test_best_las = test_las
        logger.info('Final test performance -- UAS: %.2f%%, LAS: %.2f%%' % (test_best_uas, test_best_las))


    def evaluate(self, test_data, args, vocab):
        self.parser_model.eval()

        all_gold_heads, all_gold_rels = [], []
        all_pred_heads, all_pred_rels = [], []
        with torch.no_grad():
            for batch_data in batch_iter(test_data, args.test_batch_size):
                batcher = batch_variable(batch_data, vocab, args.device)
                (bert_ids, bert_lens, bert_mask), true_heads, true_rels = batcher
                arc_score, rel_score = self.parser_model(batcher[0])
                pred_heads, pred_rels = self.decode(arc_score, rel_score, bert_lens.gt(0))
                for i, sent in enumerate(batch_data):
                    l = len(sent)
                    all_gold_heads.append(true_heads[i][1:l].tolist())
                    all_gold_rels.append(true_rels[i][1:l].tolist())
                    all_pred_heads.append(pred_heads[i][1:l].tolist())
                    all_pred_rels.append(pred_rels[i][1:l].tolist())
                
        uas, las = evaluation(all_gold_heads, all_gold_rels, all_pred_heads, all_pred_rels)
        return uas, las


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
                              torch.arange(seq_len, dtype=torch.long, device=pred_arc_score.device).unsqueeze(0),
                              pred_heads].contiguous()
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
