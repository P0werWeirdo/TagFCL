import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    # Fedarated option
    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")
    parser.add_argument('--numclass', type=int, default=10, help="number of dataset classes each task")
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--num_clients', type=int, default=10, help='initial number of clients')
    parser.add_argument('--img_size', type=int, default=32, help='')
    parser.add_argument('--save_folder', type=str, default='./save_folder',
                        help='folder name inside one_check folder')
    parser.add_argument('--epochs_local', type=int, default=20, help='local epochs of each global round')
    parser.add_argument('--epochs_server', type=int, default=20, help='server epochs of each global round')
    parser.add_argument('--log_path', type=str, default='./log/result', help='the path of result log file')

    parser.add_argument('--device', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--epochs_global', type=int, default=40, help='total number of global rounds')
    parser.add_argument('--tasks_global', type=int, default=10, help='total number of tasks')
    parser.add_argument('--tasks_epoch', type=int, default=4, help='each tasks epoch')

    parser.add_argument('--temperature', type=int, default=5, help='temperature of distillation')
    parser.add_argument('--a_c', type=float, default=0.9, help='alpha of client')
    parser.add_argument('--a_s', type=float, default=0.9, help='alpha of server')
    

    # Backtracking set
    parser.add_argument('--backtrack_region', type=int, default=3, help='region of backtracking')
    parser.add_argument('--backtrack_time', type=int, default=1, help='times of backtracking after a new task')
    parser.add_argument('--compensate_round', type=int, default=0, help='')

    # Local tasks
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')

    # Learning rate
    parser.add_argument('--lr_local', type=float, default=0.2, help='learning rate')
    parser.add_argument('--lr_server', type=float, default=0.2, help='learning rate')
    parser.add_argument('--lr_mask', type=float, default=1e-4,
                        help='Learning rate for mask')
    

    # Masking options.
    parser.add_argument('--mask_init', default='1s', choices=['1s', 'uniform', 'weight_based_1s'],
                        help='Type of mask init')
    parser.add_argument('--mask_scale', type=float, default=1e-2,
                        help='Mask initialization scaling')
    parser.add_argument('--mask_scale_gradients', type=str, default='none',
                        choices=['none', 'average', 'individual'],
                        help='Scale mask gradients by weights')
    parser.add_argument('--threshold_fn', choices=['binarizer', 'ternarizer'],
                        help='Type of thresholding function')
    parser.add_argument('--threshold', type=float, default=2e-3, help='')
    parser.add_argument('--pruning_interval', type=int, default=2, help='')
    parser.add_argument('--pruning_frequency', type=int, default=2, help='')
    parser.add_argument('--initial_sparsity', type=float, default=0.0, help='')
    parser.add_argument('--target_sparsity', type=float, default=0.1, help='')

    # Other.
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')

    args = parser.parse_args()
    return args