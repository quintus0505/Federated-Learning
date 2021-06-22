import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist
from paillier_test import enc_tensor, dec_tensor, generate_keypair
import numpy as np
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client():

    def __init__(self, args, dataset=None, idxs=None, w=None, priv=None, pub=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        self.priv = priv
        self.pub = pub

    def update_keypair(self, priv, pub):
        self.priv = priv
        self.pub = pub

    def sigmasq_func(self):
        eps_u, delta_u = self.comp_reverse()
        return 2. * np.log(1.25 / delta_u) * self.args.C ** 2 / (eps_u ** 2)

    def comp_reverse(self):
        return self.args.eps / self.args.num_users, self.args.delta / self.args.num_users

    def train(self):
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        w_new = net.state_dict()

        update_w = {}
        if self.args.mode == 'plain':
            for k in self.model.state_dict().keys():
                update_w[k] = w_new[k] - w_old[k]
            return update_w, sum(batch_loss) / len(batch_loss)

        elif self.args.mode == 'DP':
            '''1. part one DP mechanism'''
            sigmasq = self.sigmasq_func()
            for k in self.model.state_dict().keys():
                current_update_w = w_new[k] - w_old[k]
                # Clip gradient
                clipped_gradient = current_update_w / max(1, np.linalg.norm(
                    current_update_w) / self.args.C)

                update_w[k] = clipped_gradient

            return update_w, sum(batch_loss) / len(batch_loss)

        elif self.args.mode == 'Paillier':
            '''2. part two Paillier enc'''
            for k in self.model.state_dict().keys():
                update_w[k] = w_new[k] - w_old[k]

            enc_update_w = copy.deepcopy(update_w)
            # encode
            for k in w_new.keys():
                size = self.model.state_dict()[k].size()
                enc_update_w[k] = enc_tensor(self.pub, enc_update_w[k].numpy().tolist(), size)

            return enc_update_w, sum(batch_loss) / len(batch_loss)

    def update(self, w_glob):
        if self.args.mode == 'plain' or self.args.mode == 'DP':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'Paillier':
            '''2. part two Paillier dec'''
            # decode
            for k in self.model.state_dict().keys():
                size = self.model.state_dict()[k].size()
                w_glob[k] = torch.Tensor(dec_tensor(self.priv, self.pub, w_glob[k], size))

            self.model.load_state_dict(w_glob)
