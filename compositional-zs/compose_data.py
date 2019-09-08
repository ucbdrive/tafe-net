import torch
import torch.utils.data as data
from os.path import join, exists


def load_word_embeddings(emb_file, vocab):
    vocab = [v.lower() for v in vocab]

    embeds = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        embeds[line[0]] = wvec

    new_embs = []
    for k in vocab:
        if '-' in k:
            ks = k.split('-')
            emb = torch.stack([embeds[it] for it in ks]).mean(dim=0)
        else:
            emb = embeds[k]
        new_embs.append(emb)

    embeds = torch.stack(new_embs)
    return embeds


class CompositionDataset(data.Dataset):
    def __init__(self, root, phase, split='compositional-split',
                 emb_file='glove/glove.6B.300d.txt', triple=False,
                 feat='resnet18'):
        self.root = root  # default is `data/compositional-zs`
        self.phase = phase
        self.split = split
        self.emb_file = join(root, emb_file)
        self.triple = triple  # SPO triplet is used in StanfordVRD
        self.feat = feat
        feat_file = '{}/{}_features.t7'.format(root, feat)
        activation_data = torch.load(feat_file)
        self.activations = dict(
            zip(activation_data['files'], activation_data['features']))
        self.feat_dim = activation_data['features'].size(1)

        # load splits
        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = \
            self.parse_split()
        assert len(set(self.train_pairs) & set(self.test_pairs)) == 0, \
            'train and test are not mutually exclusive'

        # load pretrained word2vec embeddings
        if self.emb_file is not None:
            att_emb_file = join(root, 'attrs_embs.t7')
            if exists(att_emb_file):
                self.attrs_embds = torch.load(att_emb_file)
            else:
                self.attrs_embds = load_word_embeddings(
                    self.emb_file, self.attrs)
                torch.save(self.attrs_embds, att_emb_file)

            obj_emb_file = join(root, 'objs_embs.t7')
            if exists(obj_emb_file):
                self.objs_embds = torch.load(obj_emb_file)
            else:
                self.objs_embds = load_word_embeddings(
                    self.emb_file, self.objs)
                torch.save(self.objs_embds, obj_emb_file)
        else:
            raise NotImplementedError

        self.curr_pairs = self.train_pairs if self.phase == 'train' \
            else self.test_pairs

        self.pair2idx = {pair: idx for idx, pair in enumerate(self.curr_pairs)}

        if self.triple:
            self.embeddings = [torch.cat([
                self.objs_embds[self.objs.index(sub)],
                self.attrs_embds[self.attrs.index(verb)],
                self.objs_embds[self.objs.index(obj)]])
                for (sub, verb, obj) in self.curr_pairs]
        else:
            self.embeddings = [torch.cat([
                self.attrs_embds[self.attrs.index(attr)],
                self.objs_embds[self.objs.index(obj)]])
                for (attr, obj) in self.curr_pairs]

        self.embeddings = torch.stack(self.embeddings)

        self.data, self.labels = self.get_split_info()

    def get_split_info(self):
        data = torch.load(self.root + '/metadata.t7')
        images, targets = [], []

        for instance in data:
            if self.triple:
                image, sub, verb, obj = instance['image'], instance['sub'], \
                                        instance['pred'], instance['obj']

                key = (sub, verb, obj)
            else:
                image, attr, obj = instance['image'], instance['attr'], \
                                   instance['obj']
                key = (attr, obj)

            if key in self.curr_pairs:
                images.append(image)
                targets.append(self.curr_pairs.index(key))

        return images, targets

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))

            if self.triple:
                subs, verbs, objs = zip(*pairs)
                return subs, verbs, objs, pairs
            else:
                attrs, objs = zip(*pairs)
                return attrs, objs, pairs

        if self.triple:
            tr_subs, tr_verbs, tr_objs, tr_pairs = parse_pairs(
                '{}/{}/train_pairs.txt'.format(self.root, self.split))
            ts_subs, ts_verbs, ts_objs, ts_pairs = parse_pairs(
                '{}/{}/test_pairs.txt'.format(self.root, self.split))

            # The attributes are now the `verbs` and the subjects and objects
            # share the same label space
            all_attrs, all_objs = sorted(list(set(tr_verbs + ts_verbs))), \
                                  sorted(list(set(tr_objs + ts_objs +
                                                  tr_subs + ts_subs)))
        else:
            tr_attrs, tr_objs, tr_pairs = parse_pairs(
                '{}/{}/train_pairs.txt'.format(self.root, self.split))
            ts_attrs, ts_objs, ts_pairs = parse_pairs(
                '{}/{}/test_pairs.txt'.format(self.root, self.split))

            all_attrs, all_objs = sorted(list(set(tr_attrs + ts_attrs))), \
                                  sorted(list(set(tr_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs

    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        feat = self.activations[image]
        return feat, label

    def __len__(self):
        return len(self.data)

