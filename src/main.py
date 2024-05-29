import torch
import math
import os.path as osp
import os
from option import args_parser
from utils.Fed_utils import *
from utils.server_utils import *

import models
import models.layers as nl

from iCIFAR100 import *
from tiny_imagenet import *
from mini_imagenet import *
from tqdm import tqdm
from utils import Metric, classification_accuracy
from client import Client


def record_result(acc, epochs, args, is_backtrack=False, time=0):
    with open(args.log_path, 'a') as out_file:
        if is_backtrack:
            log_str = 'Backtrack Epochs_{}, Time_{}:\n'.format(epochs, time)
            out_file.write(log_str)
        else:
            log_str = 'Epochs_{}:\n'.format(epochs)
            out_file.write(log_str)
        
        acc = np.array(acc)
        acc_mean = np.mean(acc, axis=0)

        for task_id in range(acc_mean.shape[0]):
            log_str = 'Task_{}, Accuracy:{:.2f}'.format(task_id + 1, acc_mean[task_id] * 100)
            out_file.write(log_str + '\n')
        acc_mean = np.mean(acc_mean)
        log_str = 'Overall, Accuracy:{:.2f}'.format(acc_mean * 100)
        out_file.write(log_str + '\n')
        out_file.write('\n')
        out_file.flush()

def main():
    args = args_parser()
    num_clients = args.num_clients

    ## seed settings
    setup_seed(args.seed)

    # set server model
    server_model = models.ServerModel([], {}, args)

    # transform
    train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.24705882352941178),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    # dataset
    if args.dataset == 'cifar100':
        train_dataset = iCIFAR100('dataset', transform=train_transform, download=False)
        test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=False)
    elif args.dataset == 'tiny_imagenet':
        train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform,
                                      test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform,
                                      test_transform=test_transform)
        test_dataset.get_data()
    else:
        train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        test_dataset.get_data()

    clients = [Client(args, train_dataset, test_dataset, i) for i in range(num_clients)]

    for i in range(1, args.tasks_global + 1):
        server_model.add_dataset(i, args.numclass)

    for ep in range(args.epochs_global):
        task_id = (ep // args.tasks_epoch) + 1
        for client in clients:
            client.train_head(task_id, args.epochs_local // 2)
        messages_to_server = [ client.get_message(task_id) for client in clients]
        messages_to_client = train_and_get_message(server_model, messages_to_server, args)
        for client, msg in zip(clients, messages_to_client):
            client.train_tail(task_id, args.epochs_local // 2, msg)
        acc = []
        for client in clients:
            acc.append(client.only_validate())
        record_result(acc, ep + 1, args, is_backtrack=False)

        head_region = task_id - args.backtrack_region if task_id - args.backtrack_region > 0 else 1
        backtrack_time = args.backtrack_time
        if task_id - head_region == 0:
            continue

        for bt_time in range(backtrack_time):
            backtrack_tasks = [ _ for _ in range(head_region, task_id)]
            total_clients = [ _ for _ in range(num_clients)]
            time = 0
            acc = []
            for task_back in backtrack_tasks:
                time += 1
                this_time_clients = []
                if time != len(backtrack_tasks):
                    this_time_clients = random.sample(total_clients, max(len(total_clients) // len(backtrack_tasks), 1))
                else:
                    this_time_clients = total_clients

                total_clients = list(set(total_clients) - set(this_time_clients))

                if this_time_clients == []:
                    break
                
                for t in this_time_clients:
                    clients[t].train_head(task_back, args.epochs_local // 2)

                messages_to_server = [ clients[t].get_message(task_back) for t in this_time_clients ]
                messages_to_client = train_and_get_message(server_model, messages_to_server, args)

                for t, msg in zip(this_time_clients, messages_to_client):
                    clients[t].train_tail(task_back, args.epochs_local // 2, msg)
                for t in this_time_clients:
                    acc.append(clients[t].only_validate())
            record_result(acc, ep + 1, args, True, bt_time + 1)
            

if __name__ == '__main__':
    main()
    