""" Brancher classes. """

from collections import OrderedDict
import pyscipopt as scip
import numpy as np

import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class Brancher(scip.Branchrule):
    """
    Base class for scip.Branchrule subclasses.
    Callback method branchexeclp is customized in each subclass.
    """
    def initialize(self):
        pass

    def branchinit(self):
        pass


""" IL branchers """


class ILEvalBrancher(Brancher):
    """
    Brancher using trained Imitation Learning policy and ILEvalEnv.
    Evaluation mode is deterministic.
    """

    def __init__(self, model, device, policy, state_dims, verbose):
        super(ILEvalBrancher, self).__init__()

        self.model = model
        self.device = device
        self.policy = policy.to(device)
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        self.branch_count = 0
        self.branchexec_count = 0
        self.episode_rewards = []

    def choose(self, probs):
        if len(probs.size()) == 0:
            probs = probs.unsqueeze(0)
        confidence_score, branch_decision = probs.max(0)
        return confidence_score, branch_decision

    def branchexeclp(self, allowaddcons):

        self.branchexec_count += 1

        # get state representations
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        node_state = self.model.getNodeState(self.node_dim)
        mip_state = self.model.getMIPState(self.mip_dim)

        # torchify states
        cands_state_mat = torch.from_numpy(cands_state_mat.astype('float32')).to(self.device)
        node_state = torch.from_numpy(node_state.astype('float32')).to(self.device)
        mip_state = torch.from_numpy(mip_state.astype('float32')).to(self.device)

        # select action from the policy probs
        probs = self.policy(cands_state_mat, node_state, mip_state)
        probs = probs.squeeze()
        confidence_score, action = self.choose(probs)  # the chosen variable

        # define the SCIP branch var
        var = cands[action.item()]
        # branch on the selected variable (SCIP Variable object)
        self.model.branchVar(var)
        self.branch_count += 1

        if self.verbose:
            print('\tBranch count: {}. Selected var: {}.'.format(
                   self.branch_count, cands_pos[action.item()]))

        result = scip.SCIP_RESULT.BRANCHED
        if result == scip.SCIP_RESULT.BRANCHED:
            _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
            assert chosen_variable is not None
            assert chosen_variable.isInLP()

        return {'result': result}

    def finalize(self):
        pass

    def finalize_zero_branch(self):
        pass


""" SCIP branchers """


class SCIPCollectBrancher(Brancher):
    """
    Brancher to run SCIP data collection for imitation learning, with SCIPCollectEnv class.
    Instead of a single policy, 'explorer' and 'expert' rules are specified
    (each should be a string corresponding to a SCIP branching rule).
    The explorer policy runs for the top k branching decisions, then the expert takes over.
    Data is collected from expert decisions only.
    """
    def __init__(self, model, explorer, expert, k, state_dims, verbose):
        super(SCIPCollectBrancher, self).__init__()

        self.model = model
        self.explorer = explorer
        self.expert = expert
        self.k = k
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        # counters and data structures
        self.branchexec_count = 0
        self.branch_count = 0
        self.explore = True
        self.explorer_count = 0
        self.collect_count = 0  # data collect counter
        self.collect_dict = OrderedDict()  # data dictionary to be filled with states and labels
        self.nnodes_list = []
        self.nnodesleft_list = []

    def branchexeclp(self, allowaddcons):

        # determine whether explorer or expert should be run
        if self.branch_count < self.k:
            self.explore = True
        else:
            self.explore = False

        if self.explore:
            # branch with explorer
            assert isinstance(self.explorer, str)
            self.branchexec_count += 1
            self.nnodes_list.append(self.model.getNNodes())
            self.nnodesleft_list.append(self.model.getNNodesLeft())
            result = self.model.executeBranchRule(self.explorer, allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                self.explorer_count += 1
                self.branch_count += 1
                if self.verbose:
                    print('\tExplore count: {} (exec. {}).'.format(self.explorer_count, self.branchexec_count))
        else:

            # get state representations
            cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
            cands_state_mat.astype('float32')
            node_state = self.model.getNodeState(self.node_dim).astype('float32')
            mip_state = self.model.getMIPState(self.mip_dim).astype('float32')

            # branch with expert
            assert isinstance(self.expert, str)
            self.branchexec_count += 1
            self.nnodes_list.append(self.model.getNNodes())
            self.nnodesleft_list.append(self.model.getNNodesLeft())
            result = self.model.executeBranchRule(self.expert, allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                self.collect_count += 1
                self.branch_count += 1
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()

                self.collect_dict[self.collect_count] = {
                    'cands_state_mat': cands_state_mat,
                    'mip_state': mip_state,
                    'node_state': node_state,
                    'varLPpos': chosen_variable.getCol().getLPPos(),
                    'varRELpos': cands_pos.index(chosen_variable.getCol().getLPPos()),
                }
                if self.verbose:
                    print('\tBranch count: {} (exec. {}). '
                          'Selected varLPpos: {}. '
                          'Selected varRELpos: {}. '
                          'Num cands: {}'.format(self.branch_count, self.branchexec_count,
                                                 chosen_variable.getCol().getLPPos(),
                                                 cands_pos.index(chosen_variable.getCol().getLPPos()),
                                                 len(cands),
                                                 ))

        return {'result': result}

    def finalize(self):
        pass


class SCIPEvalBrancher(Brancher):
    """
    Brancher for SCIP evaluation run, with SCIPEvalEnv class.
    A single branching policy is specified (a string corresponding to a SCIP branching rule).
    """
    def __init__(self, model, policy, verbose):
        super(SCIPEvalBrancher, self).__init__()

        self.model = model
        self.policy = policy
        self.verbose = verbose

        # counters and data structures
        self.branchexec_count = 0
        self.branch_count = 0
        self.nnodes_list = []
        self.nnodesleft_list = []

    def branchexeclp(self, allowaddcons):

        # SCIP branching rule
        assert isinstance(self.policy, str)
        self.branchexec_count += 1
        self.nnodes_list.append(self.model.getNNodes())
        self.nnodesleft_list.append(self.model.getNNodesLeft())
        result = self.model.executeBranchRule(self.policy, allowaddcons)
        if result == scip.SCIP_RESULT.BRANCHED:
            self.branch_count += 1
            _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
            assert chosen_variable is not None
            assert chosen_variable.isInLP()

            if self.verbose:
                print('\tBranch count: {} (exec. {}).'.format(self.branch_count, self.branchexec_count))

        return {'result': result}

    def finalize(self):
        pass
