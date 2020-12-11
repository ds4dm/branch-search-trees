""" Pure-SCIP evaluations (no learning involved). """

import sys
import time
import os
from collections import OrderedDict
import yaml
import random
import argparse

import numpy as np
import pyscipopt as scip
import pickle

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

if __name__ == '__main__':

    # parser definition
    parser = argparse.ArgumentParser(description='Parser for evaluation of SCIP branching policies.')
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
        '-p',
        '--policy',
        type=str,
        default='relpscost',
        help='Name of SCIP branching rule to be used.'
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
    outfile_dir = os.path.join(paths[args.system]['out_dir'], 'SCIPEval_{}_{}_{}_{}'.format(
        args.system, args.setting, args.seed, args.policy
    ))
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir, exist_ok=True)
    instance_file_path = os.path.join(paths[args.system]['instances_dir'], args.name)  # name contains extension mps.gz
    name = args.name.split('.')[0]

    # get cutoff
    cutoff_dict = pickle.load(open(paths[args.system]['cutoff_dict'], 'rb'))
    assert name in cutoff_dict

    # setup the environment and collect data
    env = SCIPEvalEnv()
    exp_dict = env.run_episode(
        instance=instance_file_path,
        name=name,
        policy=args.policy,
        scip_seed=args.seed,
        cutoff_value=cutoff_dict[name],
        scip_limits=limits,
        scip_params=settings[args.setting],
        verbose=args.verbose,
    )

    # dump the dictionary
    f = open(os.path.join(outfile_dir, '{}_{}_{}_info.pkl'.format(name, args.seed, args.policy)), 'wb')
    pickle.dump(exp_dict, f)
    f.close()
