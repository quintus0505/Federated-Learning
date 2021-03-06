import torch
import copy
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import CNNMnist
from paillier_test import enc_tensor, dec_tensor, generate_keypair
import numpy as np


class Server():
    def __init__(self, args, w, priv=None, pub=None):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        self.priv = priv
        self.pub = pub

    def update_keypair(self, priv, pub):
        self.priv = priv
        self.pub = pub

    def sigmasq_func(self):  # Compute the variance for a (eps, delta)-DP Gaussian mechanism with sensitivity = sens
        eps_u, delta_u = self.comp_reverse()
        sigma = 2. * np.log(1.25 / delta_u) * self.args.C ** 2 / (eps_u ** 2)
        return sigma

    def comp_eps(self):
        _, delta_u = self.comp_reverse()
        return math.sqrt(2 * math.log(1.25 / delta_u)) * \
               self.args.C / self.args.sigma * self.args.num_users

    def comp_reverse(self):
        return self.args.eps / self.args.num_users, \
               self.args.delta / self.args.num_users

    def FedAvg(self):
        if self.args.mode == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]

        elif self.args.mode == 'DP':
            '''1. part one DP mechanism'''
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                    # Add noise
                    noise = np.random.normal(0, self.args.C * self.args.sigma, size=update_w_avg[k].shape)
                    update_w_avg[k] += noise

                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]

        elif self.args.mode == 'Paillier':
            '''2. part two Paillier add'''
            # decode
            for k in self.clients_update_w[0].keys():
                for i in range(0, len(self.clients_update_w)):
                    self.clients_update_w[i][k] = torch.Tensor(
                        dec_tensor(self.priv, self.pub, self.clients_update_w[i][k], self.model.state_dict()[k].size()))

            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]

                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]

        # return
        if self.args.mode == 'plain' or self.args.mode == 'DP':
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

        elif self.args.mode == 'Paillier':
            enc_state_dict = copy.deepcopy(self.model.state_dict())
            # encode
            for k in enc_state_dict:
                size = self.model.state_dict()[k].size()
                enc_state_dict[k] = enc_tensor(self.pub, enc_state_dict[k].numpy().tolist(), size)

            return enc_state_dict, sum(self.clients_loss) / len(self.clients_loss)

    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
