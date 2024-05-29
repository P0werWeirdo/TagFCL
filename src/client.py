import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter

import logging
import os
import pdb
import math
from tqdm import tqdm
import sys
import numpy as np
from manager import Manager
from utils import set_logger, remove_logger

class Client(object):
    def __init__(self, args, train_dataset, test_dataset, client_id):
        self.client_id = client_id
        self.total_tasks = args.tasks_global
        self.max_task = 0
        self.total_steps = args.tasks_epoch
        self.now_step = 0
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.log_path = 'log/client' + str(client_id)
        self.manager = Manager(args, train_dataset, test_dataset, self.log_path)
        self.args = args

        self.target_sp = [0]
        for i in range(args.tasks_global):
            remain_n = 1 - 0.1 * i
            tmp = 0
            if remain_n != 0:
                tmp = 1 - 0.1 / remain_n
                self.target_sp.append(tmp)
            else:
                tmp = 0.0
                self.target_sp.append(tmp)
        
    def train_head(self, task_id, epochs):
        set_logger(self.log_path)
        if task_id == self.max_task:
            self.train_now_task_head(task_id, epochs)
        elif task_id < self.max_task:
            self.train_past_task_head(task_id, epochs)
        else:
            self.now_step = 0
            self.max_task = task_id
            self.train_now_task_head(task_id, epochs)
        remove_logger()

    def train_past_task_head(self, task_id, epochs):
        self.manager.train_only_backtrack(task_id, epochs,self.target_sp[task_id])
    
    def train_now_task_head(self, task_id, epochs):
        self.now_step += 1
        to_prune = True
        if self.now_step == 1 or task_id == self.args.tasks_global:
            to_prune = False
        else:
            to_prune = True
            gradual_prune = self.args.tasks_epoch - 1
            initial_sparsity = (self.now_step - 2) * self.target_sp[task_id] / gradual_prune
            target_sparsity = (self.now_step - 1) * self.target_sp[task_id] / gradual_prune

        if to_prune:
            self.manager.train_only_now(task_id, epochs, True, initial_sparsity, target_sparsity - (target_sparsity - initial_sparsity) / 2)
        else:
            self.manager.train_only_now(task_id, epochs, False)
        
    def train_tail(self, task_id, epochs, message):
        assert self.client_id == message['client_id']
        set_logger(self.log_path)
        if task_id == self.max_task:
            self.train_now_task_tail(task_id, epochs, message)
        else:
            self.train_past_task_tail(task_id, epochs, message)
        remove_logger()

    def train_past_task_tail(self, task_id, epochs, message):
        self.manager.train_diss_backtrack(task_id, message, epochs, self.target_sp[task_id])
    
    def train_now_task_tail(self, task_id, epochs, message):
        to_prune = True

        if self.now_step == 1 or task_id == self.args.tasks_global:
            to_prune = False
        else:
            to_prune = True
            gradual_prune = self.args.tasks_epoch - 1
            initial_sparsity = (self.now_step - 2) * self.target_sp[task_id] / gradual_prune
            target_sparsity = (self.now_step - 1) * self.target_sp[task_id] / gradual_prune

        if to_prune:
            self.manager.train_diss_now(task_id, message, epochs, True, initial_sparsity + (target_sparsity - initial_sparsity) / 2, target_sparsity)
        else:
            self.manager.train_diss_now(task_id, message, epochs, False)
        
    
    def get_message(self, task_id):
        return self.manager.get_message(self.client_id, task_id)
               
    def only_validate(self):
        acc = []
        set_logger(self.log_path)
        for task_id in range(1, self.max_task + 1):
            acc.append(self.manager.only_validate(task_id))
        remove_logger()
        return acc