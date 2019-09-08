import os
import re
import torch
import numpy as np
import itertools
import json
from os.path import join
import glob
import pdb
import shutil


class UnNormalizer:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(l):
    return list(itertools.chain.from_iterable(l))


def adjust_learning_rate_epoch(args, optimizer, epoch):
    if args.low <= epoch <= args.high:
        lr_new = args.lr * (args.step_ratio ** 1)
        lr_base = args.blr * (args.step_ratio ** 1)
        lr_emb = args.elr * (args.step_ratio ** 1)
    elif epoch > args.high:
        lr_new = args.lr * (args.step_ratio ** 2)
        lr_base = args.blr * (args.step_ratio ** 2)
        lr_emb = args.elr * (args.step_ratio ** 2)
    else:
        lr_new = args.lr
        lr_base = args.blr
        lr_emb = args.elr

    print('Epoch [{}] new param learning rate is {}, '
          'base learning rate is {}, '
          'emb learning rate is {}'.format(epoch, lr_new, lr_base, lr_emb))
    for param_group in optimizer.param_groups:
        if 'name' in param_group:
            if param_group['name'] == 'embedding':
                param_group['lr'] = lr_emb
            elif param_group['name'] == 'feature':
                param_group['lr'] = lr_base
            else:
                param_group['lr'] = lr_new
        else:
            param_group['lr'] = lr_new


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
                    prefix='model'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(
            filename, os.path.join(save_path, prefix + '_best.pth.tar'))


def save_args(args):
    _dict = args._get_kwargs()
    with open(join(args.save_path, 'arguments.json'), 'w') as fp:
        json.dump(_dict, fp)


def save_code(args):
    os.makedirs(join(args.save_path, 'code'), exist_ok=True)
    for f in glob.glob('./models/*.py') + glob.glob('./*') + \
            glob.glob('./data/*.py') + glob.glob('./utils/*.py'):
        if not os.path.isfile(f):
            continue
        tar = join(args.save_path, 'code', os.path.basename(f))
        shutil.copy(f, tar)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def per_class_avg_accuracy(output, target, classes, topk=(1,)):
    """Computes the averaged per-class accuracy for the prediction"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float()
        res_c = []
        for c in range(classes):
            pos_c = target == c
            try:
                correct_k_c = correct_k[pos_c].sum().float()
                total_c = pos_c.sum().float()
            except:
                pdb.set_trace()
            if total_c > 0:
                res_c.append((correct_k_c / total_c).item())
        res_c = np.mean(res_c)
        res.append(res_c * 100)
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
