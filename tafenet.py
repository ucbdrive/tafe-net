import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class TAFENet(nn.Module):
    def __init__(self, meta_learner, config, feature_in_dim, feat_dim, **kwargs):
        super(TAFENet, self).__init__()
        self.config = config
        self.cin = feat_dim
        self.num_classes = 1  # use Sigmod logit
        in_dim = feature_in_dim

        # set up feature layers
        for i, c in enumerate(config):
            setattr(self, 'linear_{}'.format(i),
                    nn.Linear(in_dim, config[i], bias=True))
            in_dim = config[i]

        self.dropout = nn.Dropout(0.5, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        # binary classifier (num_classes=1 with Sigmoid outputs)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, self.num_classes))
        self._initialize_weights()

        # the meta-learner is initialized in a separate function.
        self.meta_learner = meta_learner(config, feat_dim=feat_dim, **kwargs)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, task_emb):
        weights = self.meta_learner(task_emb)
        for i in range(len(self.config)):
            x = getattr(self, 'linear_{}'.format(i))(x)
            weight = weights[i]

            # the weight factorization shown in equation (5)
            # the meta_learner only outputs task-specific weights
            x *= weight
            if i != len(self.config) - 1:
                x = self.relu(x)

        assert len(weights) == len(self.config) + 1
        # calculate the cosine_similarity between the task embedding
        # (putted it as the last element in weights) and TAFE
        dist = self.relu(F.cosine_similarity(weights[-1], x.clone()))
        weights.append(dist)

        x = self.dropout(x)
        out = self.classifier(x)

        out = torch.sigmoid(out)
        return out, weights


class MetaLearner(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, feat_dim, nhidden=3,
                 return_emb=True, **kwargs):
        super(MetaLearner, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.nhidden = nhidden
        self.return_emb = return_emb
        self.config = config  # channel config of the primary network

        # task embedding network
        module_list = [nn.Linear(self.input_dim, self.hidden_dim, bias=False),
                       nn.BatchNorm1d(self.hidden_dim),
                       nn.ReLU(inplace=True)]

        for i in range(nhidden - 2):
            module_list.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True)])

        module_list.extend([
            nn.Linear(self.hidden_dim, self.feat_dim, bias=True),
            nn.ReLU(inplace=True)])
        self.emb = nn.Sequential(*module_list)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        # gate output dim matches the feature layer size
        for i, cf in enumerate(config):
            setattr(self, 'generator_{}'.format(i),
                    nn.Linear(self.feat_dim, cf))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.emb(x)
        outputs = []
        for i in range(len(self.config)):
            out = getattr(self, 'generator_{}'.format(i))(x)
            outputs.append(out)

        if self.return_emb:
            # add the task embedding as the last output in order to calculate
            # the embedding loss more conveniently
            outputs.append(x)
        return outputs
