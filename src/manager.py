import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy as cp
import numpy as np

import logging
import models
import models.layers as nl


from prune import SparsePruner
from torch.nn.parameter import Parameter
from utils import Optimizers, set_logger, classification_accuracy, get_one_hot

from tqdm import tqdm
from utils import Metric
from torch.utils.data import DataLoader
from DistillDataset import DistillDataset

class Manager(object):
    def __init__(self, args, train_dataset, test_dataset, log_path):
        self.args = args
        self.learned_ep = [0 for _ in range(args.tasks_global + 1)]
        self.now_task = 0
        self.masks = {}
        self.shared_layer_info={}
        self.model = models.__dict__['resnet18'](dataset_history=[], dataset2num_classes={},
            network_width_multiplier=1.0, shared_layer_info={})
        
        self.init_weight_model = cp.deepcopy(self.model)
        self.new_weight_model = cp.deepcopy(self.model)
        for name, module in self.model.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                self.masks[name] = mask.cuda(self.args.device)
        self.add_task()

        self.device = args.device
        self.to_train_model = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.log_path = log_path
        self.distill_dataset = None
        self.T = args.temperature

    def train_only_now(self, task_id, epochs, prune=False, initial_sparsity=0, target_sparsity=0):
        self.model = cp.deepcopy(self.init_weight_model)
        self.model.set_dataset(task_id)
        self.set_task(task_id)
        self.model.cuda(self.args.device)

        train_loader = self._get_train_dataloader_normal([_ for _ in range( (task_id - 1) * 10, task_id * 10)],
          self.args.batch_size)
        val_loader = self._get_test_dataloader_normal([(task_id - 1) * 10, task_id * 10], self.args.batch_size)

        if prune:
            self.process(self.model, self.args, task_id, 'prune', epochs, train_loader, val_loader, initial_sparsity, target_sparsity)
        else:
            self.process(self.model, self.args, task_id, 'finetune', epochs, train_loader, val_loader)
    
    def train_diss_now(self, task_id, message, epochs, prune=False, initial_sparsity=0, target_sparsity=0):
        self.distill_dataset.add_output_logits(message['output_logits'])
        train_loader = self._get_train_dataloader_distill(self.distill_dataset, self.args.batch_size)
        val_loader = self._get_test_dataloader_normal([(task_id - 1) * 10, task_id * 10], self.args.batch_size)

        if prune:
            self.process(self.model, self.args, task_id, 'prune', epochs, train_loader, val_loader, initial_sparsity, target_sparsity, distill=True)
        else:
            self.process(self.model, self.args, task_id, 'finetune', epochs, train_loader, val_loader, distill=True)

        self.save_changes(task_id)
        self.save_params(task_id, self.model, save_to_new=True)
        self.save_params(task_id, self.model, save_to_new=False)

    def train_only_backtrack(self, task_id, epochs, target_sp):
        self.model = cp.deepcopy(self.init_weight_model)
        self.model.set_dataset(task_id)
        self.load_params(task_id, self.model)
        self.set_task(task_id)
        self.apply_mask(self.model, task_id)
        self.model.cuda(self.args.device)

        train_loader = self._get_train_dataloader_normal([_ for _ in range( (task_id - 1) * 10, task_id * 10)],
          self.args.batch_size)
        val_loader = self._get_test_dataloader_normal([(task_id - 1) * 10, task_id * 10], self.args.batch_size)
        self.process(self.model, self.args, task_id, 'prune', epochs, train_loader, val_loader, target_sp, target_sp)

    def train_diss_backtrack(self, task_id, message, epochs, target_sp):
        self.distill_dataset.add_output_logits(message['output_logits'])
        train_loader = self._get_train_dataloader_distill(self.distill_dataset, self.args.batch_size)
        val_loader = self._get_test_dataloader_normal([(task_id - 1) * 10, task_id * 10], self.args.batch_size)
        self.process(self.model, self.args, task_id, 'prune', epochs, train_loader, val_loader, target_sp, target_sp, distill=True)

        self.save_changes(task_id)
        self.save_params(task_id, self.model, save_to_new=True)  

    def only_validate(self, task_id):
        model = cp.deepcopy(self.init_weight_model)
        model.set_dataset(task_id)
        self.load_params(task_id, model)
        self.set_for_validate(model, task_id)
        self.apply_mask(model, task_id)
        model.cuda(self.args.device)

        val_loader = self._get_test_dataloader_normal([(task_id - 1) * 10, task_id * 10], self.args.batch_size)
        pruner = SparsePruner(model, self.masks, self.args, 0, 0, task_id, "inference", 0.0, 0.0)

        start_epoch = self.learned_ep[task_id]
        return self.validate(model, pruner, task_id, start_epoch, val_loader)

    def process(self, model, args, task_id, mode, epochs, train_loader, val_loader, initial_sparsity=0, target_sparsity=0, distill=False):
        curr_prune_step = begin_prune_step = len(train_loader)
        end_prune_step = args.pruning_interval * len(train_loader)

        lr = args.lr_local
        lr_mask = args.lr_mask

        named_params = dict(model.named_parameters())
        params_to_optimize_via_SGD = []
        named_of_params_to_optimize_via_SGD = []
        masks_to_optimize_via_Adam = []
        named_of_masks_to_optimize_via_Adam = []

        for name, param in named_params.items():
            if 'classifiers' in name:
                if '.{}.'.format(model.datasets.index(task_id - 1)) in name:
                    params_to_optimize_via_SGD.append(param)
                    named_of_params_to_optimize_via_SGD.append(name)
                continue
            elif 'piggymask' in name:
                masks_to_optimize_via_Adam.append(param)
                named_of_masks_to_optimize_via_Adam.append(name)
            else:
                params_to_optimize_via_SGD.append(param)
                named_of_params_to_optimize_via_SGD.append(name)

        optimizer_network = optim.SGD(params_to_optimize_via_SGD, lr=lr,
                            weight_decay=0.0, momentum=0.9, nesterov=True)
        optimizers = Optimizers()
        optimizers.add(optimizer_network, lr)

        if masks_to_optimize_via_Adam:
            optimizer_mask = optim.Adam(masks_to_optimize_via_Adam, lr=lr_mask)
            optimizers.add(optimizer_mask, lr_mask)

        """Performs training."""
        curr_lrs = []
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                curr_lrs.append(param_group['lr'])
                break
        
        pruner = SparsePruner(model, self.masks, self.args, begin_prune_step, end_prune_step, self.now_task, mode, initial_sparsity, target_sparsity)
        if mode == 'prune':
            logging.info('')
            logging.info('Before pruning: ')
            logging.info('Sparsity range: {} -> {}'.format(initial_sparsity, target_sparsity))
            logging.info('')
        elif mode == 'finetune':
            pruner.make_finetuning_mask() 
            logging.info('Finetune stage...')
        
        start_epoch = self.learned_ep[task_id]
        self.learned_ep[task_id] += epochs
        for epoch_idx in range(start_epoch, self.learned_ep[task_id]):
            avg_train_acc, curr_prune_step = self.train(model, task_id, optimizers, pruner, epoch_idx, curr_lrs, curr_prune_step, train_loader, mode, distill)
            avg_val_acc = self.validate(model, pruner, task_id, epoch_idx, val_loader)

    def train(self, model, task_id, optimizers, pruner, epoch_idx, curr_lrs, curr_prune_step, train_loader, mode, distill=False):
        model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        if distill:
            with tqdm(total=len(train_loader),
                        desc='Train Ep. #{}: '.format(epoch_idx + 1),
                        disable=False,
                        ascii=True) as t:
                for batch_idx, (indexs, data, target, logits) in enumerate(train_loader):
                    if self.args.cuda:
                        data, target, logits = data.cuda(self.device), target.cuda(self.device), logits.cuda(self.device)

                    optimizers.zero_grad()

                    output, extract_feature = model(data)
                    num = data.size(0)

                    label = torch.argmax(target, dim=1)
                    loss_true = self._compute_loss(indexs, output, label)
                    loss_KL = F.kl_div(F.log_softmax(output/self.T, dim=1), F.softmax(logits/self.T, dim=1)) * (self.T * self.T)
                    loss = loss_true * self.args.a_c + loss_KL * (1 - self.args.a_c) 
                    loss.backward(retain_graph=True)

                    train_loss.update(loss.cpu(), num)

                    train_accuracy.update(classification_accuracy(output, label), num)

                    pruner.do_weight_decay_and_make_grads_zero()

                    optimizers.step()

                    if mode == 'prune' and pruner.initial_sparsity != pruner.target_sparsity:
                        curr_prune_step += 1
                        pruner.gradually_prune(curr_prune_step)

                    if self.now_task == 1:
                        t.set_postfix({'loss': train_loss.avg.item(),
                                        'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                                        'lr': curr_lrs[0],
                                        'sparsity': pruner.calculate_sparsity()})
                    else:
                        t.set_postfix({'loss': train_loss.avg.item(),
                                        'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                                        'lr': curr_lrs[0],
                                        'sparsity': pruner.calculate_sparsity()})
                    t.update(1)
        else:
            with tqdm(total=len(train_loader),
                        desc='Train Ep. #{}: '.format(epoch_idx + 1),
                        disable=False,
                        ascii=True) as t:
                for batch_idx, (indexs, data, target) in enumerate(train_loader):
                    target = target - 10 * (task_id - 1)
                    if self.args.cuda:
                        data, target = data.cuda(self.device), target.cuda(self.device)

                    optimizers.zero_grad()

                    output, extract_feature = model(data)
                    num = data.size(0)

                    loss = self._compute_loss(indexs, output, target)

                    train_loss.update(loss.cpu(), num)
                    loss.backward()

                    train_accuracy.update(classification_accuracy(output, target), num)

                    pruner.do_weight_decay_and_make_grads_zero()

                    # Gradient is applied across all ranks
                    optimizers.step()

                    # Set pruned weights to 0.
                    if mode == 'prune' and pruner.initial_sparsity != pruner.target_sparsity:
                        curr_prune_step += 1
                        pruner.gradually_prune(curr_prune_step)

                    if self.now_task == 1:
                        t.set_postfix({'loss': train_loss.avg.item(),
                                        'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                                        'lr': curr_lrs[0],
                                        'sparsity': pruner.calculate_sparsity()})
                    else:
                        t.set_postfix({'loss': train_loss.avg.item(),
                                        'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                                        'lr': curr_lrs[0],
                                        'sparsity': pruner.calculate_sparsity()})
                    t.update(1)
        
        summary = {'loss': '{:.3f}'.format(train_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                   'lr': curr_lrs[0],
                   'sparsity': '{:.3f}'.format(pruner.calculate_sparsity())
                   }

        if self.log_path:
            logging.info(('In baseline_cifar100_acc.txt()-> Train Ep. #{}, Task: {} '.format(epoch_idx + 1, task_id)
                         + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return train_accuracy.avg.item(), curr_prune_step

    # the main process to validate
    def validate(self, model, pruner, task_id, epoch_idx, val_loader):
        """Performs evaluation."""
        model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(val_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for setp, (indexs, data, target) in enumerate(val_loader):
                    target = target - 10 * (task_id - 1)
                    if self.args.cuda:
                        data, target = data.cuda(self.device), target.cuda(self.device)

                    logits, extract_feature = model(data)
                    num = data.size(0)
                    val_loss.update(self._compute_loss(indexs, logits, target).cpu(), num)
                    val_accuracy.update(classification_accuracy(logits, target), num)

                    if task_id == 1:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': pruner.calculate_sparsity(),
                                       'task{} ratio'.format(task_id): pruner.calculate_curr_task_ratio(),
                                       'zero ratio': pruner.calculate_zero_ratio()})
                    else:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': pruner.calculate_sparsity(),
                                       'task{} ratio'.format(task_id): pruner.calculate_curr_task_ratio(),
                                       'shared_ratio': pruner.calculate_shared_part_ratio(),
                                       'zero ratio': pruner.calculate_zero_ratio()})
                    t.update(1)

        summary = {'loss': '{:.3f}'.format(val_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                   'sparsity': '{:.3f}'.format(pruner.calculate_sparsity()),
                   'task{} ratio'.format(task_id): '{:.3f}'.format(pruner.calculate_curr_task_ratio()),
                   'zero ratio': '{:.3f}'.format(pruner.calculate_zero_ratio())}
        if task_id != 1:
            summary['shared_ratio'] = '{:.3f}'.format(pruner.calculate_shared_part_ratio())

        if self.log_path:
            logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
                         + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return val_accuracy.avg.item()

    def get_message(self, client_id, task_id):
        extracted_feature_list = []
        logits_list = []
        label_list = []
        extracted_feature_list_test = []
        labels_list_test = []

        data_to_msg = []

        msg_loader = self._get_train_dataloader_msg([_ for _ in range( (task_id - 1) * 10, task_id * 10)],
          self.args.batch_size)
        test_loader = self._get_test_dataloader_normal([(task_id - 1) * 10, task_id * 10], self.args.batch_size)
        self.model.eval()

        with torch.no_grad():
            for setp, (indexs, data, target) in enumerate(msg_loader):
                target = target - 10 * (task_id - 1)
                if self.args.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                logits, extract_feature = self.model(data)
                target = get_one_hot(target, self.args.numclass, self.device)

                data_to_msg.extend(list(data.cpu()))
                extracted_feature_list.extend(list(extract_feature.cpu()))
                logits_list.extend(list(logits.cpu()))
                label_list.extend(list(target.cpu()))

        with torch.no_grad():
            for setp, (indexs, data, target) in enumerate(test_loader):
                target = target - 10 * (task_id - 1)
                if self.args.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)
                logits, extract_feature = self.model(data)
                target = get_one_hot(target, self.args.numclass, self.device)

                extracted_feature_list_test.extend(list(extract_feature.cpu()))
                labels_list_test.extend(list(target.cpu()))

        self.distill_dataset = DistillDataset(data_to_msg, label_list)

        return {
            "client_id": client_id,
            "task_id": task_id,
            "extracted_feature_list": extracted_feature_list,
            "logits_list": logits_list,
            "label_list": label_list,
            "extracted_feature_list_test": extracted_feature_list_test,
            "labels_list_test": labels_list_test
        }
    
    def _compute_loss(self, indexs, output, label):
        target = get_one_hot(label, self.args.numclass, self.device)
        output, target = output.cuda(self.device), target.cuda(self.device)
        loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
        return loss_cur

    def _get_train_dataloader_normal(self,train_classes, batchsize):
        self.train_dataset.getTrainData(train_classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=batchsize,
                                  # num_workers=8,
                                  pin_memory=True)
        return train_loader

    def _get_train_dataloader_msg(self, train_classes, batchsize):
        self.train_dataset.getTrainData(train_classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=False,
                                  batch_size=batchsize,
                                  # num_workers=8,
                                  pin_memory=True)
        return train_loader

    def _get_train_dataloader_distill(self, train_dataset, batchsize):
        train_loader = DataLoader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=batchsize,
                                  # num_workers=8,
                                  pin_memory=True)
        return train_loader
    
    def _get_test_dataloader_normal(self, test_classes, batchsize):
        self.test_dataset.getTestData(test_classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                  shuffle=False,
                                  batch_size=batchsize,
                                  # num_workers=4,
                                  pin_memory=True)
        return test_loader
    def save_changes(self, task_id):
        """Save changes to shared_layer_info"""
        for name, module in self.model.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                if module.bias is not None:
                    self.shared_layer_info[task_id][
                        'bias'][name] = module.bias.detach().clone()
                if module.piggymask is not None:
                    self.shared_layer_info[task_id][
                        'piggymask'][name] = module.piggymask.detach().clone()
            elif isinstance(module, nn.BatchNorm2d):
                self.shared_layer_info[task_id][
                    'bn_layer_running_mean'][name] = module.running_mean.detach().clone()
                self.shared_layer_info[task_id][
                    'bn_layer_running_var'][name] = module.running_var.detach().clone()
                self.shared_layer_info[task_id][
                    'bn_layer_weight'][name] = module.weight.detach().clone()
                self.shared_layer_info[task_id][
                    'bn_layer_bias'][name] = module.bias.detach().clone()
            elif isinstance(module, nn.PReLU):
                self.shared_layer_info[task_id][
                    'prelu_layer_weight'][name] = module.weight.detach().clone()

    def add_task(self):
        for task_id in range(0, self.args.tasks_global + 1):
            self.model.add_dataset(task_id, self.args.numclass)
            self.init_weight_model.add_dataset(task_id, self.args.numclass)
            self.new_weight_model.add_dataset(task_id, self.args.numclass)

            if task_id not in self.shared_layer_info:
                self.shared_layer_info[task_id] = {
                    'bias': {},
                    'bn_layer_running_mean': {},
                    'bn_layer_running_var': {},
                    'bn_layer_weight': {},
                    'bn_layer_bias': {},
                    'piggymask': {}
                }

                piggymasks = {}
                if task_id > 1:
                    for name, module in self.model.named_modules():
                        if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                            piggymasks[name] = torch.zeros_like(self.masks[name], dtype=torch.float32)
                            piggymasks[name].fill_(0.01)
                            piggymasks[name] = Parameter(piggymasks[name])
                self.shared_layer_info[task_id]['piggymask'] = piggymasks

    def apply_mask(self, model, task_id):
        """To be done to retrieve weights just for a particular dataset."""
        for name, module in model.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                weight = module.weight.data
                mask = self.masks[name].cuda()
                # zeros = torch.eq(mask, 0)
                # print(zeros.sum())
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(task_id)] = 0.0
        return

    def save_params(self, task_id, model, save_to_new=True):
        target_model = self.new_weight_model if save_to_new else self.init_weight_model
        if self.args.cuda:
            model.cuda(self.args.device)
            target_model.cuda(self.args.device)
        for (name1, module1), (name2, module2) in zip(model.named_modules(), target_model.named_modules()):
            if isinstance(module1, nl.SharableConv2d) or isinstance(module1, nl.SharableLinear):
                weight = module1.weight.data.detach().clone()
                if name1 != name2:
                    print("fk line:354")
                module2.weight.data[self.masks[name1].eq(task_id)] = weight[self.masks[name1].eq(task_id)].detach().clone()
        target_model.copy_dataset(task_id, model)
        
    def load_params(self, task_id, model):
        if self.args.cuda:
            model.cuda(self.args.device)
            self.new_weight_model.cuda(self.args.device)
        for (name1, module1), (name2, module2) in zip(model.named_modules(), self.new_weight_model.named_modules()):
            if isinstance(module1, nl.SharableConv2d) or isinstance(module1, nl.SharableLinear):
                weight = module1.weight.data
                weight[self.masks[name1].eq(task_id)] = module2.weight.data[self.masks[name1].eq(task_id)].detach().clone()
        self.new_weight_model.set_dataset(task_id)
        model.copy_dataset(task_id, self.new_weight_model)
        model.set_dataset(task_id)

    def set_task(self, task_id):
        self.now_task = task_id

        if task_id <= len(self.shared_layer_info):
            piggymasks = self.shared_layer_info[task_id]['piggymask']
        else:
            print('Manager: set_task error')

        if task_id > 1:
            for name, module in self.model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    module.piggymask = piggymasks[name]
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    module.piggymask = None
        
        if self.shared_layer_info[task_id]['bn_layer_running_mean'] != {}:
            for name, module in self.model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[task_id]['bias'][name]

                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[task_id][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[task_id][
                        'bn_layer_running_var'][name]
                    module.weight = Parameter(self.shared_layer_info[task_id][
                        'bn_layer_weight'][name])
                    module.bias = Parameter(self.shared_layer_info[task_id][
                        'bn_layer_bias'][name])

                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[task_id][
                        'prelu_layer_weight'][name]
        elif task_id > 1:
            task_id -= 1
            for name, module in self.model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[task_id]['bias'][name].detach().clone()

                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[task_id][
                        'bn_layer_running_mean'][name].detach().clone()
                    module.running_var = self.shared_layer_info[task_id][
                        'bn_layer_running_var'][name].detach().clone()
                    module.weight = Parameter(self.shared_layer_info[task_id][
                        'bn_layer_weight'][name].detach().clone())
                    module.bias = Parameter(self.shared_layer_info[task_id][
                        'bn_layer_bias'][name].detach().clone())

                elif isinstance(module, nn.PReLU):
                    module.weight = Parameter(self.shared_layer_info[task_id][
                        'prelu_layer_weight'][name].detach().clone())
            task_id += 1
        #     print('no bias')
  
    def set_for_validate(self, model, task_id):
        if task_id <= len(self.shared_layer_info):
            piggymasks = self.shared_layer_info[task_id]['piggymask']
        else:
            print('Manager: set_task error')

        if task_id > 1:
            for name, module in model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    module.piggymask = piggymasks[name]
        else:
            for name, module in model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    module.piggymask = None
        assert self.shared_layer_info[task_id]['bn_layer_running_mean'] != {}
        if self.shared_layer_info[task_id]['bn_layer_running_mean'] != {}:
            for name, module in model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[task_id]['bias'][name]

                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[task_id][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[task_id][
                        'bn_layer_running_var'][name]
                    module.weight = Parameter(self.shared_layer_info[task_id][
                        'bn_layer_weight'][name])
                    module.bias = Parameter(self.shared_layer_info[task_id][
                        'bn_layer_bias'][name])

                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[task_id][
                        'prelu_layer_weight'][name]
