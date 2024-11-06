# Abstract Class for different pruning implementations to follow

import torch
import torch.nn.utils.prune as prune
import transformers
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

class Pruner(ABC):
    @abstractmethod
    def __init__(self, baseline, layers, strengths):
        """ Initializer for 

        Args:
            baseline (torch.nn.Module): Baseline model (0% sparsity)
            layers (list[torch.nn]): Valid pruning layers in model
            strengths (list): What the pruning strength of each model should be.
                              Number of models to prune inferred from this.
            NOTE: The sparsity will NOT equal the pruning strength if you are only
                  pruning a subset of the layers: p_stength >= sparsity
        """

        # Map model to sparsity
        self.baseline = baseline
        self.strengths = strengths
        self.layers = layers
        self.models = [] # Keep on CPU to avoid VRAM problems
        self.num_models = len(strengths)

    @abstractmethod
    def prune(self):
        pass

    @staticmethod
    def buildParameterList(model, layers):
        # =[torch.nn.Linear]
        """ Compile a list of parameters (weights) that can be pruned.
            Allows the user to select a subset of the network.
            Note: BERT's self attention layer's include linear layers, hence
            they are selected as the default.

        Returns:
            list: List of valid parameters to prune
        """
        parameters_to_prune = []
        for _, module in model.named_modules():
            for layer in layers:
                if isinstance(module, layer):
                    parameters_to_prune.append((module, 'weight'))
        return parameters_to_prune

    @staticmethod
    def computeSparsity(model):
        """ Compute the current sparsity of the network.
            Done by counting 0's, with the assumption that 
            active weights are non-zero.

        Args:
            model (torch.nn.module): model to calculate sparsity wrt
        """
        zeros = 0
        elements = 0
        for _, param in model.named_parameters():

            # Count number of pruned params
            zero_count = torch.sum(param==0.0).item()
            zeros += zero_count
            # Count total number of params
            element_count = param.numel()
            elements += element_count

        if(elements == 0):
            return 0

        return zeros / elements

