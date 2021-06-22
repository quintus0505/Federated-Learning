import numpy as np
# from numpy.lib.function_base import gradient
import torch
from torchvision import datasets, transforms, utils
from models.Nets import CNNMnist
from options import args_parser
from paillier_test import enc, dec, generate_keypair, enc_add, enc_add_const, enc_mul_const
from client import *
from server import *
import copy


def load_dataset():
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test


def create_client_server():
    priv, pub = generate_keypair(1024)
    num_items = int(len(dataset_train) / args.num_users)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]
    net_glob = CNNMnist(args=args).to(args.device)

    # 平分训练数据，i.i.d.
    # 初始化同一个参数的模型
    for i in range(args.num_users):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - new_idxs)
        new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()), priv=priv,pub=pub)
        clients.append(new_client)

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()), priv=priv, pub=pub)

    return clients, server


if __name__ == '__main__':
    print(torch.cuda.is_available())

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)

    print("load dataset...")
    dataset_train, dataset_test = load_dataset()

    print("clients and server initialization...")
    clients, server = create_client_server()
    eps = server.comp_eps()
    if server.args.mode == 'plain' or server.args.mode == 'Paillier':
        print("mode: {}".format(server.args.mode))
    elif server.args.mode == 'DP':
        print(
            "mode: {} eps: {} sigma: {} delta: {} C: {}".format(server.args.mode, eps, server.args.sigma, server.args.delta,
                                                                server.args.C))

    # training
    print("start training...")
    for iter in range(args.epochs):
        priv, pub = generate_keypair(1024)
        server.update_keypair(priv, pub)
        server.clients_update_w, server.clients_loss = [], []
        for idx in range(args.num_users):
            clients[idx].update_keypair(priv, pub)
            update_w, loss = clients[idx].train()
            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)

        # calculate global weights
        w_glob, loss_glob = server.FedAvg()

        # update local weights
        for idx in range(args.num_users):
            clients[idx].update(w_glob)

        # print loss
        acc_train, loss_train = server.test(dataset_train)
        acc_test, loss_test = server.test(dataset_test)
        print('Round {:3d}, Training average loss {:.3f}'.format(iter, loss_glob))
        print("Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))

    # testing

    acc_train, loss_train = server.test(dataset_train)
    acc_test, loss_test = server.test(dataset_test)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
