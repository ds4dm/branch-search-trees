""" Imitation Learning (IL) evaluations, using SCIP. """

import sys
import time
import os
import yaml
import random
import argparse
import pickle
from collections import OrderedDict

import numpy as np
import torch
import pyscipopt as scip

from src.environments import *
from models.feedforward import *

import faulthandler
faulthandler.enable()

# solver parametric setting, key ('sandbox' or 'default') to be specified in argparse --setting
settings = {
    'sandbox': {
        'heuristics': False,        # enable primal heuristics
        'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
        'conflict_usesb': False,    # use SB conflict analysis
        'probing_bounds': False,    # use probing bounds identified during SB
        'checksol': False,
        'reevalage': 0,
    },
    'default': {
        'heuristics': True,
        'cutoff': False,
        'conflict_usesb': True,
        'probing_bounds': True,
        'checksol': True,
        'reevalage': 10,
    },
}

# limits in solvers
limits = {
    'node_limit': -1,
    'time_limit': 3600.,
}

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
    parser = argparse.ArgumentParser(description='Parser for IL evaluation experiments.')
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help='Pathway to torch checkpoint to be loaded.'
    )
    parser.add_argument(
        '--cutoff_dict',
        type=str,
        help='Pathway to pickled dictionary containing cutoff values.'
    )
    parser.add_argument(
        '--instances_dir',
        type=str,
        help='Pathway to the MILP instances.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Pathway to save all the SCIP eval pickle files.'
    )
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        help='Name of the MILP instance.mps.gz (containing extension) to be processed.'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help='Random seed for SCIP solver.'
    )
    parser.add_argument(
        '--setting',
        type=str,
        default='sandbox',
        help='Solver parameters setting.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='Flag on verbosity.'
    )
    args = parser.parse_args()

    # load a checkpoint path (cpu load of a gpu checkpoint)
    chkpnt = torch.load(args.checkpoint, map_location='cpu')
    print('Checkpoint loaded from path {}...'.format(args.checkpoint))

    # read config from checkpoint: the policy parameters are inferred from the checkpoint args
    checkpoint_args = chkpnt['args']

    # set all random seeds
    scip_seed = args.seed

    instance_file_path = os.path.join(args.instances_dir, args.name)  # name contains extension mps.gz
    name = args.name.split('.')[0]

    # set device (cpu for eval)
    device = torch.device('cpu')

    # get cutoff
    cutoff_dict = pickle.load(open(args.cutoff_dict, 'rb'))
    assert name in cutoff_dict

    # setup the environment and a policy within it
    env = ILEvalEnv(device=device)

    if checkpoint_args.policy_type == 'TreeGatePolicy':
        policy = TreeGatePolicy(
            var_dim=state_dims['var_dim'],
            node_dim=state_dims['node_dim'],
            mip_dim=state_dims['mip_dim'],
            hidden_size=checkpoint_args.hidden_size,
            depth=checkpoint_args.depth,
            dropout=checkpoint_args.dropout,
            dim_reduce_factor=checkpoint_args.dim_reduce_factor,
            infimum=checkpoint_args.infimum,
            norm=checkpoint_args.norm,
        )
        # set the policy into eval mode
        policy.eval()
        # load the policy parameters
        policy.load_state_dict(chkpnt['state_dict'])
        policy = policy.to(device)
        policy_name = 'TreeGatePolicy'
    elif checkpoint_args.policy_type == 'NoTreePolicy':
        policy = NoTreePolicy(
            var_dim=state_dims['var_dim'],
            node_dim=state_dims['node_dim'],
            mip_dim=state_dims['mip_dim'],
            hidden_size=checkpoint_args.hidden_size,
            depth=checkpoint_args.depth,
            dropout=checkpoint_args.dropout,
            dim_reduce_factor=checkpoint_args.dim_reduce_factor,
            infimum=checkpoint_args.infimum,
            norm=checkpoint_args.norm,
        )
        # set the policy into eval mode
        policy.eval()
        # load the policy parameters
        policy.load_state_dict(chkpnt['state_dict'])
        policy = policy.to(device)
        policy_name = 'NoTreePolicy'
    else:
        raise ValueError('A valid policy should be set.')

    # main evaluation
    eps = np.finfo(np.float32).eps.item()
    with torch.no_grad():
        exp_dict = env.run_episode(
            instance=instance_file_path,
            name=args.name.split('.')[0],
            policy=policy,
            policy_name=policy_name,
            state_dims=state_dims,
            scip_seed=args.seed,
            cutoff_value=cutoff_dict[name],
            scip_limits=limits,
            scip_params=settings[args.setting],
            verbose=args.verbose,
        )

    # dump the exp_dict
    f = open(os.path.join(args.out_dir, '{}_{}_ILEval_info.pkl'.format(name, args.seed)), 'wb')
    pickle.dump(exp_dict, f)
    f.close()
