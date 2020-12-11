""" Collect MILP data for Imitation Learning (IL), by performing SCIP roll-outs. """

import os
import yaml
import random
import argparse
import pickle
from collections import OrderedDict
import numpy as np
import pyscipopt as scip

from src.environments import *

import faulthandler
faulthandler.enable()


# system-specific paths, key 'MYSYSTEM' to be specified in argparse --system
paths = {
    'MYSYSTEM': {
        'out_dir': '',          # path to output directory
        'instances_dir': '',    # path to MILP instances
        'cutoff_dict': '',      # path to pickled dictionary containing cutoff values
    },
}

# solver parametric setting, key ('sandbox' or 'default') to be specified in argparse --setting
settings = {
    'sandbox': {
        'heuristics': False,        # enable primal heuristics
        'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
        'conflict_usesb': False,    # use SB conflict analysis
        'probing_bounds': False,    # use probing bounds identified during SB
        'checksol': False,          # check LP solutions found during strong branching with propagation
        'reevalage': 0,             # number of intermediate LPs solved to trigger reevaluation of SB value
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

# collection branching rules
collectors = {
    'explorer': 'random',
    'expert': 'relpscost',
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
    parser = argparse.ArgumentParser(description='Parser for SCIP data collection.')
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
        '-k',
        '--k_nodes',
        type=int,
        default=10,
        help='Number of initial nodes to be explored randomly, before starting data collection.'
    )
    parser.add_argument(
        '--setting',
        type=str,
        default='sandbox',
        help='Solver parameters setting.'
    )
    parser.add_argument(
        '--system',
        type=str,
        default='gz_local',
        help='System on which script is run.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='Flag on verbosity.'
    )
    args = parser.parse_args()

    # setup output directory and path to instance
    outfile_dir = os.path.join(paths[args.system]['out_dir'], 'SCIPCollect_{}_{}_{}_{}'.format(
        args.system, args.setting, args.seed, args.k_nodes
    ))
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir, exist_ok=True)
    instance_file_path = os.path.join(paths[args.system]['instances_dir'], args.name)  # name contains extension mps.gz
    name = args.name.split('.')[0]

    # get cutoff
    cutoff_dict = pickle.load(open(paths[args.system]['cutoff_dict'], 'rb'))
    assert name in cutoff_dict

    # setup the environment and collect data
    env = SCIPCollectEnv()
    exp_dict, collect_dict = env.run_episode(
        instance=instance_file_path,
        name=name,
        explorer=collectors['explorer'],
        expert=collectors['expert'],
        k=args.k_nodes,
        state_dims=state_dims,
        scip_seed=args.seed,
        cutoff_value=cutoff_dict[name],
        scip_limits=limits,
        scip_params=settings[args.setting],
        verbose=args.verbose,
    )

    # dump the dictionaries
    f = open(os.path.join(outfile_dir, '{}_{}_{}_info.pkl'.format(name, args.seed, args.k_nodes)), 'wb')
    pickle.dump(exp_dict, f)
    f.close()

    ff = open(os.path.join(outfile_dir, '{}_{}_{}_data.pkl'.format(name, args.seed, args.k_nodes)), 'wb')
    pickle.dump(collect_dict, ff)
    ff.close()
