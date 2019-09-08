import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.checkpoint

import argparse
import logging
import models
from .zsl_data import ZSData
from utils import *
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', default='d3r_vgg11_awa2')
    parser.add_argument('--config-name', '-cfg', type=str, default='',
                        help='configuration name')
    parser.add_argument('--cmd', choices=['train', 'test'],
                        type=str, default='train')
    parser.add_argument('--gzsl', action='store_true',
                        help='generalized zero-shot learning')

    parser.add_argument('--resume', default='', type=str,
                        help='path to the checkpoint to begin with')
    parser.add_argument('--save-folder', default='zero-shot/checkpoints',
                        type=str, help='folder to save everything')
    parser.add_argument('--dataset', default='awa2',
                        choices=['awa2', 'cub', 'sun', 'awa1', 'apy'],
                        help='dataset')
    parser.add_argument('--data-root', default='data/zero-shot',
                        type=str, help='path to the data and split info')
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='batch size')
    parser.add_argument('--num-workers', default=16, type=int,
                        help='number of workers')
    parser.add_argument('--emb-dim', type=int, default=85,
                        help='input dim of the embedding network')
    parser.add_argument('--input-dim', type=int, default=2048,
                        help='input image feature dimension')
    parser.add_argument('--feat-dim', type=int, default=2048,
                        help='feature dimension.')
    parser.add_argument('--hidden-dim', type=int, default=2048,
                        help='hidden unit size of the meta learner')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate for new parameters')
    parser.add_argument('--elr', default=1e-5, type=float,
                        help='initial learning rate for embed parameters')
    parser.add_argument('--blr', default=1e-4, type=float,
                        help='initial learning rate for base parameters ')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='starting epoch')
    parser.add_argument('--epochs', default=60, type=int,
                        help='total number of training epochs')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='step ratio')
    parser.add_argument('--low', type=int, default=30,
                        help='lower threshold')
    parser.add_argument('--high', type=int, default=60,
                        help='high thresholds')
    parser.add_argument('--beta', type=float, default=1,
                        help='coefficient for the embedding loss')
    args = parser.parse_args()

    # save directory
    args.save_path = join(
        args.save_folder, args.dataset, args.arch, args.config_name)
    os.makedirs(args.save_path, exist_ok=True)

    args.run_id = len(os.listdir(args.save_path)) + 1
    args.save_path = join(args.save_path, 'run_{}'.format(args.run_id))
    os.makedirs(args.save_path, exist_ok=True)

    args.logger_file = os.path.join(
        args.save_path, 'log_{}.txt'.format(args.cmd))

    save_args(args)
    return args


def main():
    args = parse_args()

    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d " \
             "%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format=FORMAT,
                        handlers=handlers)

    logging.info('Results will be saved to {}'.format(args.save_path))
    if args.cmd == 'train':
        logging.info('start training model {}'.format(args.arch))
        run_training(args)
    elif args.cmd == 'test':
        logging.info('start testing model {}'.format(args.arch))
        test(args)
    else:
        raise NotImplementedError


def run_training(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.__dict__[args.arch](args.emb_dim, args.feat_dim,
                                       args.input_dim, args.hidden_dim)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    feat_params, classifier_params, gate_params, emb_params = [], [], [], []
    for n, p in model.named_parameters():
        if 'emb' in n:
            emb_params.append(p)
        elif 'gate' in n:
            gate_params.append(p)
        elif 'classifier' in n:
            classifier_params.append(p)
        else:
            feat_params.append(p)

    param_group = [
        {'params': emb_params, 'name': 'embedding'},
        {'params': gate_params, 'name': 'gate'},
        {'params': classifier_params, 'name': 'classifier'},
        {'params': feat_params, 'name': 'feature'}
    ]

    best_prec1 = 0
    best_prec1_u, best_prec1_s = 0, 0
    if args.resume:
        logging.info('=> loading checkpoint `{}`'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
            args.resume, checkpoint['epoch']))

    logging.info('loading the {} dataset'.format(args.dataset))
    args.data_path = join(args.data_root, args.dataset)
    logging.info('start creating train loader ...')
    trainset = ZSData(root=args.data_path, subset='train')

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    # using the validation set
    logging.info('start creating test loader ...')
    if args.gzsl:
        testset_unseen = ZSData(
            root=args.data_path, subset='test', gzsl=True, seen=False)
        test_loader_unseen = torch.utils.data.DataLoader(
            testset_unseen, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)

        testset_seen = ZSData(
            root=args.data_path, subset='test', gzsl=True, seen=True)

        test_loader_seen = torch.utils.data.DataLoader(
            testset_seen, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
    else:
        testset = ZSData(root=args.data_path, subset='test',
                         gzsl=False)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(param_group, args.lr)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate_epoch(args, optimizer, epoch)
        train(args, train_loader, model, criterion, optimizer, epoch)

        if args.gzsl:
            avg_prec1_u = validate(args, test_loader_unseen, model)

            avg_prec1_s = validate(args, test_loader_seen, model)

            avg_prec1_h = (2 * avg_prec1_s * avg_prec1_u) / (
                avg_prec1_s + avg_prec1_u)
            avg_prec1 = avg_prec1_h
            logging.info('Epoch [{}] Val: * Summary: \n'
                         '* GZSL @Unseen: {:.4f} \t '
                         '@Seen: {:.4f} \t @mean_h: {:.4f}'
                         '\n'.format(epoch, avg_prec1_u, avg_prec1_s,
                                     avg_prec1_h))
        else:
            avg_prec1 = validate(args, test_loader, model)
            logging.info('Epoch [{}] Val: * Summary: \n'
                         '* ZSL @top1: {:.4f}\n'.format(epoch, avg_prec1))

        # remember best prec@1 and save checkpoint
        is_best = avg_prec1 > best_prec1
        best_prec1 = max(avg_prec1, best_prec1)
        if is_best:
            if args.gzsl:
                best_prec1_u = avg_prec1_u
                best_prec1_s = avg_prec1_s
                logging.info('* best so far is '
                             '* GZSL @Unseen: {:.4f} \t '
                             '@Seen: {:.4f} \t @mean_h: {:.4f}\n'.format(
                    avg_prec1_u, avg_prec1_s, avg_prec1))
            else:
                logging.info('* best so far is '
                             '* ZSL @top1: {:.4f}\n'.format(avg_prec1))

        checkpoint_path = os.path.join(
            args.save_path, 'checkpoint_latest.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)

    if args.gzsl:
        logging.info("** Best GZSL: mean_h = {:.4f}\t"
                     "@Unseen = {:.4f}\t@Seen = {:.4f}".format(
            best_prec1, best_prec1_u, best_prec1_s))
    else:
        logging.info("** Best ZSL accuracy = {:.4f}".format(best_prec1))


def weighted_mse_loss(input, target, weight):
    return (torch.sum(weight * (input - target) ** 2, 1) / weight.sum(1)).mean()


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_dist = AverageMeter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    end = time.time()
    with tqdm(total=len(train_loader), ascii=True, desc='train') as t:
        for i, (input, targets, task_embs, mtarget) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # re-batch every epoch, sort of the episode way of training
            input = input.float().to(device)

            unique_lab_batch = torch.unique(mtarget)
            nclass = len(unique_lab_batch)
            rebatch_target = []
            for m in mtarget:
                rebatch_target.append((m == unique_lab_batch).nonzero()[0])
            rebatch_target = torch.cat(rebatch_target).to(device)

            # concatenate the input and output
            size = input.size()
            input_rep = input.unsqueeze(2).repeat(1, 1, nclass).transpose(1, 2)
            input_rep = input_rep.contiguous().view(-1, size[1])

            # select all the embeddings used in the current batch
            task_emb = torch.stack([task_embs[j.item()][0].float().to(device)
                                    for j in unique_lab_batch], 0)
            task_emb = task_emb.repeat(size[0], 1)

            out, decisions = model(input_rep, task_emb)
            out = out.view(-1, nclass)

            label_onehot = torch.zeros(size[0], nclass).to(device).scatter_(
                1, rebatch_target.long().view(-1, 1), 1)
            loss = criterion(out, label_onehot)

            # add embedding loss
            dist = decisions[-1]
            weight = torch.ones(label_onehot.size()).to(device)
            weight[label_onehot == 1] = nclass - 1
            dist_reshape = dist.view(-1, nclass)
            loss_dist = weighted_mse_loss(dist_reshape, label_onehot, weight)
            loss += args.beta * loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            prec1, = accuracy(out, rebatch_target, topk=(1,))
            losses.update(loss.item(), size[0])
            top1.update(prec1.item(), size[0])
            losses_dist.update(loss_dist.item(), size[0])
            t.set_postfix(loss=losses.avg, emb_loss=losses_dist.avg,
                          top1=top1.avg)
            t.update()


def validate(args, test_loader, model):
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ntasks = len(test_loader.dataset.task_list)
    logging.info('Evaluation: {} novel tasks'.format(ntasks))
    top1 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(test_loader), ascii=True, desc='test') as t:
            for i, (input, targets, task_onehots, mtarget) in enumerate(
                    test_loader):
                input, mtarget = input.float().to(device), mtarget.to(device)
                # concatenate the input and output
                size = input.size()
                input_rep = input.unsqueeze(2).repeat(
                    1, 1, ntasks).transpose(1, 2)
                input_rep = input_rep.contiguous().view(-1, size[1])

                task_emb = torch.stack([task_onehots[j][0].float().to(device)
                                        for j in range(ntasks)], 0)
                task_emb = task_emb.repeat(size[0], 1)

                out, decisions = model(input_rep, task_emb)
                out = out.view(-1, ntasks)

                prec1, = per_class_avg_accuracy(out, mtarget, ntasks, topk=(1,))
                top1.update(prec1.item(), size[0])
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                t.set_postfix(prec1=prec1.item())
                t.update()

    avg_prec1 = top1.avg
    return avg_prec1


def test(args):
    # create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.__dict__[args.arch](args.emb_dim, args.feat_dim,
                                       args.input_dim, args.hidden_dim)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)

    if args.resume:
        logging.info('=> loading checkpoint `{}`'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
            args.resume, checkpoint['epoch']))

    args.data_path = join(args.data_root, args.dataset)
    if args.gzsl:
        logging.info('Evaluating generalized zsl setting .. ')
        testset = ZSData(
            root=args.data_path, subset='test', gzsl=True, seen=False)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        avg_prec1_u = validate(args, test_loader, model)

        logging.info('evaluating seen category ...')
        testset = ZSData(root=args.data_path, subset='test',
                         gzsl=True, seen=True)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        avg_prec1_s = validate(args, test_loader, model)

        avg_prec1_h = (2 * avg_prec1_s * avg_prec1_u) / (
            avg_prec1_s + avg_prec1_u)

        logging.info('* Summary: GZSL @Unseen: {:.4f}\t@Seen: {:.4f}\t'
                     '@mean_h: {:.4f}'.format(
            avg_prec1_u, avg_prec1_s, avg_prec1_h))
    else:
        logging.info('Evaluating classic zero-shot learning setting ...')
        testset = ZSData(
            root=args.data_path, subset='test',
            gzsl=False, seen=False)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        avg_prec1_zsl, probs, decisions = validate(
            args, test_loader, model)
        logging.info('* Summary: Classic ZSL '
                     '@top1 = {:.4f}%'.format(avg_prec1_zsl))


if __name__ == '__main__':
    main()
