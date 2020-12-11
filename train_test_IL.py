""" Imitation Learning (IL) training. """

import time
import os
import random
import argparse
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import math

from src.environments import *
from models.feedforward import *
from utilities.hdf5_dataloader import *


# state dimensions
# var_dim is the dimension of each candidate variable's input, i.e., the fixed dimension of matrix C_t
# Tree_t is given by concatenation of two states, for a total dimension node_dim + mip_dim
state_dims = {
    'var_dim': 25,
    'node_dim': 8,
    'mip_dim': 53
}


if __name__ == '__main__':

    # parser definition
    parser = argparse.ArgumentParser(description='Parser for IL training experiments.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for IL training experiment.'
    )
    parser.add_argument(
        '--policy_type',
        type=str,
        choices=['NoTreePolicy', 'TreeGatePolicy'],
        help='Type of policy to use.'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        help='Hidden size of the branching policy network.'
    )
    parser.add_argument(
        '--depth',
        type=int,
        help='Depth of the branching policy network.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout parameter for the branching policy network.'
    )
    parser.add_argument(
        '--dim_reduce_factor',
        type=int,
        default=2,
        help='Dimension reduce factor of the branching policy network.'
    )
    parser.add_argument(
        '--infimum',
        type=int,
        default=8,
        help='Infimum parameter of the branching policy network.'
    )
    parser.add_argument(
        '--norm',
        type=str,
        default='none',
        help='Normalization type of the branching policy network.'
    )
    parser.add_argument(
        '--train_h5_path',
        type=str,
        help='Pathway to the train H5 file.'
    )
    parser.add_argument(
        '--val_h5_path',
        type=str,
        help='Pathway to the val H5 file.'
    )
    parser.add_argument(
        '--test_h5_path',
        type=str,
        help='Pathway to the test H5 file.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Directory to save the experimental results.'
    )
    parser.add_argument(
        '--use_gpu',
        default=False,
        action='store_true',
        help='Use gpu or not.'
    )
    parser.add_argument(
        '--opt',
        default='adam',
        type=str,
        help='Type of optimizer to use.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate.'
    )
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum optimization parameter.'
    )
    parser.add_argument(
        '--weight_decay',
        default=1e-5,
        type=float,
        help='Weight decay optimization parameter.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=40,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--lr_decay_schedule',
        type=int,
        nargs='+',
        default=[20, 30],
        help='Learning rate decay schedule.'
    )
    parser.add_argument(
        '--lr_decay_factor',
        type=float,
        default=0.1,
        help='LR decay factor.'
    )
    parser.add_argument(
        '--train_batchsize',
        type=int,
        default=32,
        help='Training batchsize.')
    parser.add_argument(
        '--eval_batchsize',
        type=int,
        default=500,
        help='Evaluation batchsize.')
    parser.add_argument(
        '--top_k',
        type=int,
        nargs='+',
        default=[2, 3, 5, 10],
        help='In addition to top-1 generalization accuracy, we track top-k.'
    )
    args = parser.parse_args()
    print(args)

    # set all the random seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # setup output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    # use gpu or cpu
    if args.use_gpu:
        import torch.backends.cudnn as cudnn

        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # if final checkpoint exists exit the script
    chkpnt_path = os.path.join(args.out_dir, 'final_checkpoint.pth.tar')
    if os.path.isfile(chkpnt_path):
        print('Final checkpoint exists, experiment has already been run, exiting...')
        exit()
    elif os.path.isfile(os.path.join(args.out_dir, 'final_job_crashed_checkpoint.pth.tar')):
        print('Experiment previously crashed, exiting...')
        exit()

    # load a checkpoint path
    chkpnt_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')
    if os.path.isfile(chkpnt_path):
        chkpnt = torch.load(chkpnt_path)
        epoch_start = chkpnt['epoch']
        print('Checkpoint loaded from path {}, starting at epoch {}...'.format(chkpnt_path, epoch_start))
    else:
        chkpnt = None
        epoch_start = 0

    # setup the policy
    if args.policy_type == 'NoTreePolicy':
        policy = NoTreePolicy(
            var_dim=state_dims['var_dim'],
            node_dim=state_dims['node_dim'],
            mip_dim=state_dims['mip_dim'],
            hidden_size=args.hidden_size,
            depth=args.depth,
            dropout=args.dropout,
            dim_reduce_factor=args.dim_reduce_factor,
            infimum=args.infimum,
            norm=args.norm,
        )
        policy_name = 'NoTreePolicy'
    elif args.policy_type == 'TreeGatePolicy':
        policy = TreeGatePolicy(
            var_dim=state_dims['var_dim'],
            node_dim=state_dims['node_dim'],
            mip_dim=state_dims['mip_dim'],
            hidden_size=args.hidden_size,
            depth=args.depth,
            dropout=args.dropout,
            dim_reduce_factor=args.dim_reduce_factor,
            infimum=args.infimum,
            norm=args.norm,
        )
        policy_name = 'TreeGatePolicy'
    else:
        raise ValueError('A valid policy should be set.')

    # set the policy into train mode
    policy.train()
    policy = policy.to(device)

    if args.opt == 'adam':
        optimizer = optim.Adam(
            policy.parameters(),
            lr=args.lr,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay
        )
        eps = np.finfo(np.float32).eps.item()
    else:
        raise ValueError('A valid optimizer should be set.')

    # specify a learning rate scheduler
    if args.lr_decay_schedule:
        lr_decay_schedule = args.lr_decay_schedule
        lr_decay_factor = args.lr_decay_factor
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_schedule, lr_decay_factor)
        use_scheduler = True
    else:
        use_scheduler = False

    # if checkpoint available, load the policy's and the optimizers parameters
    if chkpnt:
        policy.load_state_dict(chkpnt['state_dict'])
        optimizer.load_state_dict(chkpnt['optimizer'])
        scheduler.load_state_dict(chkpnt['scheduler'])
        exp_dict = chkpnt['exp_dict']
    else:
        # setup an experiment log
        exp_dict = OrderedDict()
        exp_dict['train_loss_epoch_arr'] = np.zeros(args.num_epochs, )
        exp_dict['val_loss_epoch_arr'] = np.zeros(args.num_epochs, )
        exp_dict['val_acc_epoch_arr'] = np.zeros(args.num_epochs, )
        for k in args.top_k:
            exp_dict['val_acc_top_{}_epoch_arr'.format(k)] = np.zeros(args.num_epochs, )
        exp_dict['val_nan_counter_epoch_arr'] = np.zeros(args.num_epochs, )
        exp_dict['test_loss'] = 0.0
        exp_dict['test_acc'] = 0.0
        for k in args.top_k:
            exp_dict['test_acc_top_{}'.format(k)] = 0.0
        exp_dict['test_nan_counter'] = 0

    # setup a train loader
    train_h5 = dataset_h5(
        h5_file=args.train_h5_path,
        node_dim=state_dims['node_dim'],
        mip_dim=state_dims['mip_dim'],
        var_dim=state_dims['var_dim']
    )
    num_train_batches = train_h5.__len__() // args.train_batchsize
    train_loader = DataLoader(
        dataset=train_h5,
        batch_size=args.train_batchsize,
        shuffle=True,
        collate_fn=collate_fn
    )

    # setup a val loader
    val_h5 = dataset_h5(
        h5_file=args.val_h5_path,
        node_dim=state_dims['node_dim'],
        mip_dim=state_dims['mip_dim'],
        var_dim=state_dims['var_dim']
    )
    num_val_batches = val_h5.__len__() // args.eval_batchsize
    val_loader = DataLoader(
        dataset=val_h5,
        batch_size=args.eval_batchsize,
        shuffle=False,
        collate_fn=collate_fn
    )

    # setup a test loader
    test_h5 = dataset_h5(
        h5_file=args.test_h5_path,
        node_dim=state_dims['node_dim'],
        mip_dim=state_dims['mip_dim'],
        var_dim=state_dims['var_dim']
    )
    num_test_batches = test_h5.__len__() // args.eval_batchsize
    test_loader = DataLoader(
        dataset=test_h5,
        batch_size=args.eval_batchsize,
        shuffle=False,
        collate_fn=collate_fn
    )

    # setup the loss
    criterion = nn.CrossEntropyLoss().to(device)

    # main training loop
    print('Starting training loop...\n')
    for i in range(epoch_start, args.num_epochs):
        # set the policy into train mode
        policy.train()
        start_time = time.time()
        start_time_process = time.process_time()

        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            for idx, data_tuple in enumerate(batch):
                target, node, mip, grid = data_tuple
                target, node, mip, grid = target.to(device), node.to(device), mip.to(device), grid.to(device)
                logits = policy(grid, node, mip)
                logits = logits.transpose(1, 0)
                batch_loss += criterion(logits, target)
            batch_loss /= float(args.train_batchsize)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
        running_loss /= float(num_train_batches)

        if use_scheduler:
            scheduler.step()

        train_time = time.time() - start_time
        train_time_process = time.process_time() - start_time_process
        exp_dict['train_loss_epoch_arr'][i] = running_loss

        # set the policy into eval mode
        policy.eval()
        eval_start = time.time()
        eval_start_process = time.process_time()

        total_correct = 0
        top_k_correct = dict.fromkeys(args.top_k)
        for k in args.top_k:
            top_k_correct[k] = 0
        val_acc_top_k = dict.fromkeys(args.top_k)
        total_loss = 0.0
        nan_counter = 0

        with torch.no_grad():
            for batch in val_loader:
                for idx, data_tuple in enumerate(batch):
                    target, node, mip, grid = data_tuple
                    target, node, mip, grid = target.to(device), node.to(device), mip.to(device), grid.to(device)
                    logits = policy(grid, node, mip)
                    logits = logits.transpose(1, 0)
                    _loss = criterion(logits, target).item()
                    if math.isnan(_loss):
                        nan_counter += 1
                    else:
                        total_loss += _loss
                        _, predicted = torch.max(logits, 1)
                        total_correct += predicted.eq(target.item()).cpu().item()
                        grid_size = grid.size(0)
                        for k in args.top_k:
                            max_k = min(k, grid_size)  # Accounts for when grid_size is smaller than top_k
                            top_k_correct[k] += int(target.item() in logits.topk(max_k, dim=1).indices)
            if nan_counter < val_h5.n_data:
                val_loss = total_loss / float(val_h5.n_data - nan_counter)
                val_acc = total_correct / float(val_h5.n_data - nan_counter)
                for k in args.top_k:
                    val_acc_top_k[k] = top_k_correct[k] / float(val_h5.n_data - nan_counter)
            else:
                val_loss = np.nan
                val_acc = np.nan
                for k in args.top_k:
                    val_acc_top_k[k] = np.nan

                print('Model overflow on entire val set, hyperparameter configuration is ill-posed, killing the job.')

                # save the final checkpoint
                torch.save(
                    {'state_dict': policy.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'exp_dict': exp_dict,
                     'args': args,
                     'state_dims': state_dims
                     },
                    os.path.join(args.out_dir, 'final_job_crashed_checkpoint.pth.tar')
                )

                # exit
                exit()

            eval_time = time.time() - eval_start
            eval_time_process = time.process_time() - eval_start_process
            exp_dict['val_loss_epoch_arr'][i] = val_loss
            exp_dict['val_acc_epoch_arr'][i] = val_acc
            for k in args.top_k:
                exp_dict['val_acc_top_{}_epoch_arr'.format(k)][i] = val_acc_top_k[k]
            exp_dict['val_nan_counter_epoch_arr'][i] = nan_counter

        print(
            "[Epoch {:d}] Train loss: {:.4f}. Train time: {:.2f}sec. "
            "Val loss: {:.4f}. Val acc: {:.2f}%, Val acc top-{}: {:.2f}%, Val acc top-{}: {:.2f}%. Val time: {:.2f}sec.".format(
                i + 1, running_loss, train_time, val_loss, 100 * val_acc, args.top_k[0],
                100 * val_acc_top_k[args.top_k[0]],
                args.top_k[1], 100 * val_acc_top_k[args.top_k[1]], eval_time))

        # checkpoint
        torch.save(
            {'epoch': i + 1,
             'state_dict': policy.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'exp_dict': exp_dict,
             'args': args,
             'state_dims': state_dims
             },
            os.path.join(args.out_dir, 'checkpoint.pth.tar')
        )

        # create per epoch save
        torch.save(
            {'epoch': i + 1,
             'state_dict': policy.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'exp_dict': exp_dict,
             'args': args,
             'state_dims': state_dims
             },
            os.path.join(args.out_dir, 'epoch_{}_checkpoint.pth.tar'.format(i + 1))
        )

    # put the policy into eval/validation mode
    print('\nEvaluating on the test set...\n')
    policy.eval()

    total_correct = 0
    top_k_correct = dict.fromkeys(args.top_k)
    for k in args.top_k:
        top_k_correct[k] = 0
    test_acc_top_k = dict.fromkeys(args.top_k)
    total_loss = 0.0
    nan_counter = 0

    with torch.no_grad():
        for batch in test_loader:
            for idx, data_tuple in enumerate(batch):
                target, node, mip, grid = data_tuple
                target, node, mip, grid = target.to(device), node.to(device), mip.to(device), grid.to(device)
                logits = policy(grid, node, mip)
                logits = logits.transpose(1, 0)
                _loss = criterion(logits, target).item()
                if math.isnan(_loss):
                    nan_counter += 1
                else:
                    total_loss += _loss
                    _, predicted = torch.max(logits, 1)
                    total_correct += predicted.eq(target.item()).cpu().item()
                    grid_size = grid.size(0)
                    for k in args.top_k:
                        max_k = min(k, grid_size)  # Accounts for when grid_size is smaller than top_k
                        top_k_correct[k] += int(target.item() in logits.topk(max_k, dim=1).indices)
        if nan_counter < test_h5.n_data:
            test_loss = total_loss / float(test_h5.n_data - nan_counter)
            test_acc = total_correct / float(test_h5.n_data - nan_counter)
            for k in args.top_k:
                test_acc_top_k[k] = top_k_correct[k] / float(test_h5.n_data - nan_counter)
        else:
            test_loss = np.nan
            test_acc = np.nan
            for k in args.top_k:
                test_acc_top_k[k] = np.nan

        exp_dict['test_loss'] = test_loss
        exp_dict['test_acc'] = test_acc
        for k in args.top_k:
            exp_dict['test_acc_top_{}'.format(k)] = test_acc_top_k[k]
        exp_dict['test_nan_counter'] = nan_counter

    print('Test loss: {:.6f}, Test acc: {:.2f}%, Test acc top-{}: {:.2f}%, Test acc top-{}: {:.2f}%'.format(
        test_loss, 100 * test_acc, args.top_k[0], 100 * test_acc_top_k[args.top_k[0]], args.top_k[1],
        100 * test_acc_top_k[args.top_k[1]]))

    # save the final checkpoint
    torch.save(
        {'state_dict': policy.state_dict(),
         'optimizer': optimizer.state_dict(),
         'scheduler': scheduler.state_dict(),
         'exp_dict': exp_dict,
         'args': args,
         'state_dims': state_dims
         },
        os.path.join(args.out_dir, 'final_checkpoint.pth.tar')
    )
