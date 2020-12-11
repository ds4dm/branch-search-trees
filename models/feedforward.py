""" Definitions of the IL models NoTree and TreeGate. """

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.modules import *


class NoTreePolicy(nn.Module):
    """
    NoTree policy.
    """
    def __init__(self, var_dim, node_dim, mip_dim, hidden_size, depth=2, dropout=0.0, dim_reduce_factor=2, infimum=8,
                 norm='none'):
        """
        :param var_dim: int, dimension of variable state
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param hidden_size: int, hidden size parameter for the branching policy network
        :param depth: int, depth parameter for the branching policy network
        :param dropout: float, dropout parameter for the branching policy network
        :param dim_reduce_factor: int, Dimension reduce factor of the branching policy network
        :param infimum: int, infimum parameter of the branching policy network
        :param norm: str, normalization type of the branching policy network
        """
        super(NoTreePolicy, self).__init__()
        self.dropout = dropout
        self.norm = norm
        norm_layer = get_norm_layer(norm)

        # define the dimensionality of the features and the hidden states
        self.var_dim = var_dim
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.hidden_size = hidden_size
        self.depth = depth

        # define CandidateEmbeddingNet
        self.CandidateEmbeddingNet = [nn.Linear(var_dim, hidden_size)]
        self.CandidateEmbeddingNet = nn.Sequential(*self.CandidateEmbeddingNet)

        # define the BranchingNet:
        unit_count = infimum
        input_dim = hidden_size
        self.n_layers = 1
        while unit_count < hidden_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1
        self.BranchingNet = []
        for i in range(self.n_layers):
            output_dim = int(input_dim / dim_reduce_factor)
            if i < self.n_layers - 1:
                self.BranchingNet += [nn.Linear(input_dim, output_dim),
                                      norm_layer(output_dim),
                                      nn.ReLU(True)]
            elif i == self.n_layers - 1:
                self.BranchingNet += [nn.Linear(input_dim, output_dim)]
            input_dim = output_dim
        self.BranchingNet = nn.Sequential(*self.BranchingNet)

        # do the Xavier initialization for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(tensor=m.weight, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cands_state_mat, node_state=None, mip_state=None):
        # go through the CandidateEmbeddingNet
        cands_state_mat = self.CandidateEmbeddingNet(cands_state_mat)

        # go through the BranchingNet
        cands_state_mat = self.BranchingNet(cands_state_mat)  # No TreeState input to BranchingNet
        cands_prob = cands_state_mat.mean(dim=1, keepdim=True)  # Keep the axis

        return cands_prob


class TreeGatePolicy(nn.Module):
    """
    TreeGate policy.
    """
    def __init__(self, var_dim, node_dim, mip_dim, hidden_size, depth=2, dropout=0.0, dim_reduce_factor=2, infimum=8,
                 norm='none'):
        """
        :param var_dim: int, dimension of variable state
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param hidden_size: int, hidden size parameter for the network
        :param depth: int, depth parameter for the network
        :param dropout: float, dropout parameter for the network
        :param dim_reduce_factor: int, Dimension reduce factor of the network
        :param infimum: int, infimum parameter of the network
        :param norm: str, normalization type of the network
        """
        super(TreeGatePolicy, self).__init__()
        self.dropout = dropout
        self.norm = norm

        # define the dimensionality of the features and the hidden states
        self.var_dim = var_dim
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.hidden_size = hidden_size
        self.depth = depth

        # define CandidateEmbeddingNet
        self.CandidateEmbeddingNet = [nn.Linear(var_dim, hidden_size)]
        self.CandidateEmbeddingNet = nn.Sequential(*self.CandidateEmbeddingNet)

        # define the TreeGateBranchingNet
        self.TreeGateBranchingNet = TreeGateBranchingNet(hidden_size, node_dim + mip_dim, dim_reduce_factor,
                                                         infimum, norm, depth, hidden_size)

        # do the Xavier initialization for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(tensor=m.weight, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cands_state_mat, node_state, mip_state):
        # go through the CandidateEmbeddingNet
        cands_state_mat = self.CandidateEmbeddingNet(cands_state_mat)

        # go through the TreeGateBranchingNet
        cands_prob = self.TreeGateBranchingNet(cands_state_mat, node_state, mip_state)

        return cands_prob
