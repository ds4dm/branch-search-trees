""" Environment classes, to manage the interface between learning and solver. """

import numpy as np
import os
import time
import torch
import pyscipopt as scip
from collections import OrderedDict

import src.utilities as utilities
from .branchers import *


class ILEvalEnv:
    """
    Environment to evaluate a trained Imitation Learning policy, using ILEvalBrancher.
    The specified branching policy is a trained IL policy.
    """
    def __init__(self, device):
        self.device = device

    def run_episode(self, instance, name, policy, policy_name, state_dims,
                    scip_seed, cutoff_value, scip_limits, scip_params, verbose, brancher_name='ILEvalBrancher'):
        """
        :param instance: str, pathway to instance.mps.gz
        :param name: str, name of the instance (w/o extension)
        :param policy: a trained IL policy
        :param policy_name: str, name of the policy
        :param state_dims: dict, of state dimensionalities
        :param scip_seed: int, SCIP solver seed
        :param cutoff_value: float, cutoff
        :param scip_limits: dict, specifying SCIP parameter limits
        :param scip_params: dict, specifying SCIP parameter setting
        :param verbose: bool, verbosity
        :param brancher_name: str, name of the brancher to be defined
        :return:
            exp_dict: dict, containing basic statistics on the experiment (run)
        """

        print("\nRunning IL evaluation on instance {}".format(name))
        m = scip.Model()

        # set static solver setting (scip seed and cutoff are set below)
        utilities.init_params(m, scip_limits, scip_params)

        # set scip parameters as needed (wrt the current episode setting)
        m.setBoolParam('randomization/permutevars', True)
        m.setIntParam('randomization/permutationseed', scip_seed)  # SCIP default at 0

        m.readProblem(instance)

        if scip_params['cutoff']:
            assert cutoff_value is not None
            m.setObjlimit(cutoff_value)

        # define brancher
        brancher = ILEvalBrancher(
            model=m,
            device=self.device,
            policy=policy,
            state_dims=state_dims,
            verbose=verbose,
        )
        m.includeBranchrule(
            brancher,
            name=brancher_name,
            desc="bla",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1
        )

        # perform the episode
        try:
            t0 = time.time()
            t0_process = time.process_time()
            m.optimize()
            t1_process = time.process_time()
            t1 = time.time()
            print("\tInstance: {}. Nnodes: {}. Branch count: {}. Status: {}. Gap: {:.4f}".format(
                name,
                m.getNNodes(),
                brancher.branch_count,
                m.getStatus(),
                m.getGap())
            )
        except:
            print("\tSCIP exception or error.")
            t0 = time.time()
            t0_process = time.process_time()
            t1 = t0
            t1_process = t0_process

        # update exp_dict
        exp_dict = {
            'name': name,
            'policy': policy_name,
            'seed': scip_seed,
            'nnodes': m.getNNodes(),
            'fair_nnodes': m.getFairNNodes(bytes(brancher_name, 'utf-8')),  # needs bytes encoding
            'nnodes_left': m.getNNodesLeft(),
            'nLP_iterations': m.getNLPIterations(),
            'max_depth': m.getMaxDepth(),
            'status': m.getStatus(),
            'gap': m.getGap(),
            'primal_bound': m.getPrimalbound(),
            'dual_bound': m.getDualbound(),
            'primaldualintegral': m.getPrimalDualIntegral(),
            'scip_solve_time': m.getSolvingTime(),
            'scip_presolve_time': m.getPresolvingTime(),
            'opt_time_process': t1_process - t0_process,
            'opt_time_wallclock': t1 - t0,
        }

        m.freeProb()

        return exp_dict


class SCIPCollectEnv:
    """
    Environment to run SCIP data collection for imitation learning, with SCIPCollectBrancher class.
    Instead of a single policy, 'explorer' and 'expert' rules are specified
    (each should be a string corresponding to a SCIP branching rule).
    The explorer policy runs for the top k branching decisions, then the expert takes over.
    Data is collected from expert decisions only.
    """

    def __init__(self):
        pass

    def run_episode(self, instance, name, explorer, expert, k, state_dims,
                    scip_seed, cutoff_value, scip_limits, scip_params, verbose, brancher_name='SCIPCollectBrancher'):
        """
        :param instance: str, pathway to instance.mps.gz
        :param name: str, name of the instance (w/o extension)
        :param explorer: str, SCIP branching rule to be used as explorer
        :param expert: str, SCIP branching rule to be used as expert
        :param k: int, number of branching decision to be explored before data collection
        :param state_dims: dict, of state dimensionalities
        :param scip_seed: int, SCIP solver seed
        :param cutoff_value: float, cutoff
        :param scip_limits: dict, specifying SCIP parameter limits
        :param scip_params: dict, specifying SCIP parameter setting
        :param verbose: bool, verbosity
        :param brancher_name: str, name of the brancher to be defined
        :return:
            exp_dict: dict, containing basic statistics on the experiment (run)
            brancher.collect_dict: dict, of data (states, labels) collected by the expert
        """

        print("\nRunning data collection on instance {}".format(name))
        m = scip.Model()

        # set static solver setting (scip seed and cutoff are set below)
        utilities.init_params(m, scip_limits, scip_params)

        # set scip parameters as needed (wrt the current episode setting)
        m.setBoolParam('randomization/permutevars', True)
        m.setIntParam('randomization/permutationseed', scip_seed)  # SCIP default at 0

        m.readProblem(instance)

        if scip_params['cutoff']:
            assert cutoff_value is not None
            m.setObjlimit(cutoff_value)

        brancher = SCIPCollectBrancher(
            model=m,
            explorer=explorer,
            expert=expert,
            k=k,
            state_dims=state_dims,
            verbose=verbose
        )
        m.includeBranchrule(
            brancher,
            name=brancher_name,
            desc="bla",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1
        )

        # optimize, i.e., perform the solve
        t0 = time.time()
        t0_process = time.process_time()
        m.optimize()
        t1_process = time.process_time()
        t1 = time.time()

        print("\tInstance {}. SCIP time: {} (wall-clock: {}). Nnodes: {}. FairNNodes: {}. Collected: {}".format(
            name, m.getSolvingTime(), t1 - t0, m.getNNodes(),
            m.getFairNNodes(bytes(brancher_name, 'utf-8')), brancher.collect_count
        ))

        # store episode_data
        exp_dict = {
            'name': name,
            'explorer': explorer,
            'expert': expert,
            'k': k,
            'seed': scip_seed,
            'nnodes': m.getNNodes(),
            'fair_nnodes': m.getFairNNodes(bytes(brancher_name, 'utf-8')),  # needs bytes encoding
            'nnodes_left': m.getNNodesLeft(),
            'nLP_iterations': m.getNLPIterations(),
            'max_depth': m.getMaxDepth(),
            'status': m.getStatus(),
            'gap': m.getGap(),
            'primal_bound': m.getPrimalbound(),
            'dual_bound': m.getDualbound(),
            'primaldualintegral': m.getPrimalDualIntegral(),
            'scip_solve_time': m.getSolvingTime(),
            'scip_presolve_time': m.getPresolvingTime(),
            'opt_time_process': t1_process - t0_process,
            'opt_time_wallclock': t1 - t0,
            'nnodes_list': brancher.nnodes_list,
            'nnodesleft_list': brancher.nnodesleft_list,
        }

        m.freeProb()

        return exp_dict, brancher.collect_dict


class SCIPEvalEnv:
    """
    Environment for SCIP evaluation runs, with SCIPEvalBrancher class.
    A single branching policy is specified (a string corresponding to a SCIP branching rule).
    """

    def __init__(self):
        pass

    def run_episode(self, instance, name, policy,
                    scip_seed, cutoff_value, scip_limits, scip_params, verbose, brancher_name='SCIPEvalBrancher'):
        """
        :param instance: str, pathway to instance.mps.gz
        :param name: str, name of the instance (w/o extension)
        :param policy: str, SCIP branching rule to be used
        :param scip_seed: int, SCIP solver seed
        :param cutoff_value: float, cutoff
        :param scip_limits: dict, specifying SCIP parameter limits
        :param scip_params: dict, specifying SCIP parameter setting
        :param verbose: bool, verbosity
        :param brancher_name: str, name of the brancher to be defined
        :return:
            exp_dict: dict, containing basic statistics on the experiment (run)
        """
        print("\nRunning SCIP evaluation on instance {}".format(name))
        m = scip.Model()

        # set static solver setting (scip seed and cutoff are set below)
        utilities.init_params(m, scip_limits, scip_params)

        # set scip parameters as needed (wrt the current episode setting)
        m.setBoolParam('randomization/permutevars', True)
        m.setIntParam('randomization/permutationseed', scip_seed)  # SCIP default at 0

        m.readProblem(instance)

        if scip_params['cutoff']:
            assert cutoff_value is not None
            m.setObjlimit(cutoff_value)

        brancher = SCIPEvalBrancher(
            model=m,
            policy=policy,
            verbose=verbose
        )
        m.includeBranchrule(
            brancher,
            name=brancher_name,
            desc="bla",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1
        )

        # optimize, i.e., perform the solve
        t0 = time.time()
        t0_process = time.process_time()
        m.optimize()
        t1_process = time.process_time()
        t1 = time.time()

        print("\tInstance {}. SCIP time: {} (wall-clock: {}). Nnodes: {}. FairNNodes: {}".format(
            name, m.getSolvingTime(), t1 - t0, m.getNNodes(), m.getFairNNodes(bytes(brancher_name, 'utf-8'))
        ))

        # store episode_data
        exp_dict = {
            'name': name,
            'policy': policy,
            'seed': scip_seed,
            'nnodes': m.getNNodes(),
            'fair_nnodes': m.getFairNNodes(bytes(brancher_name, 'utf-8')),  # needs bytes encoding
            'nnodes_left': m.getNNodesLeft(),
            'nLP_iterations': m.getNLPIterations(),
            'max_depth': m.getMaxDepth(),
            'status': m.getStatus(),
            'gap': m.getGap(),
            'primal_bound': m.getPrimalbound(),
            'dual_bound': m.getDualbound(),
            'primaldualintegral': m.getPrimalDualIntegral(),
            'scip_solve_time': m.getSolvingTime(),
            'scip_presolve_time': m.getPresolvingTime(),
            'opt_time_process': t1_process - t0_process,
            'opt_time_wallclock': t1 - t0,
            'nnodes_list': brancher.nnodes_list,
            'nnodesleft_list': brancher.nnodesleft_list,
        }

        m.freeProb()

        return exp_dict
