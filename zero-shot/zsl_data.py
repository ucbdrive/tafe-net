from scipy.io import loadmat
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
import numpy as np
import os
from os.path import join, exists
import pickle
import pdb


class ZSData(data.Dataset):
    def __init__(self, root, subset='train', gzsl=False, seen=False,
                 target_transform=None):
        super(ZSData, self).__init__()
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform
        self.subset = subset
        self.gzsl = gzsl
        self.seen = seen

        # load mat
        self.att_splits = loadmat(join(root, 'att_splits.mat'))

        if 'original_att' in self.att_splits:
            self.att = self.att_splits['original_att'].T
        else:
            print('using att instead of original_att')
            self.att = self.att_splits['att'].T

        # load features, label or raw images
        self.feats_labels = loadmat(join(root, 'res101.mat'))
        self.feats = self.feats_labels['features'].T
        self.labels = self.feats_labels['labels'].astype(int).squeeze() - 1

        if self.subset == 'train':
            self.loc = self.att_splits['trainval_loc'].squeeze() - 1
        elif self.subset == 'val':
            self.loc = self.att_splits['test_unseen_loc'].squeeze() - 1
        else:
            if self.gzsl:
                if self.seen:
                    self.loc = self.att_splits['test_seen_loc'].squeeze() - 1
                else:
                    self.loc = self.att_splits['test_unseen_loc'].squeeze() - 1
            else:
                self.loc = self.att_splits['test_unseen_loc'].squeeze() - 1

        self.subset_labels = self.labels[self.loc]
        if self.gzsl:
            self.task_list = np.unique(self.labels)
        else:
            self.task_list = np.unique(self.subset_labels)

        self.subset_feats = self.feats[self.loc]

    def __getitem__(self, index):
        img, target = self.subset_feats[index], self.subset_labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        # provide a list of task
        labels, att_embs = [], []
        for t in self.task_list:
            label = int(target == t)
            att_emb = self.att[t]
            labels.append(label)
            att_embs.append(att_emb)
        new_target = self.task_list.tolist().index(target)
        return img, labels, att_embs, new_target

    def __len__(self):
        return len(self.subset_labels)
