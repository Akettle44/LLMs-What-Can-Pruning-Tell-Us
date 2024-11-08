# File containing classes for valid pruning strategies

# So far: L1

import numpy as np
import torch.nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from overrides import override
from .prune_interface import Pruner

class L1Pruner(Pruner):
    """ Pruning using L1 Unstructured Pruning
    """

    @override
    def __init__(self, baseline, save_dir, layers=[torch.nn.Linear], strengths=list(np.linspace(0, 0.9, 10))):
        # Init base variables + set pruning method
        super().__init__(baseline, save_dir, layers, strengths)
        self.pruning_method = prune.L1Unstructured

    @override
    def prune(self):
        """ Prune the baseline model via specified strengths
        """

        # Don't re-prune already pruned models
        if len(self.models) == len(self.strengths):
            return

        # Each p is a particular pruning strength [0, 1]
        for p in self.strengths:
            # Deep copy model
            model_i = deepcopy(self.baseline)
            pruning_parameters = Pruner.buildParameterList(model_i, self.layers)

            if p > 0:
                # Perform pruning
                prune.global_unstructured(
                    pruning_parameters,
                    pruning_method=self.pruning_method,
                    amount=p
                )

                # Make pruning permanent
                for module,name in pruning_parameters:
                    prune.remove(module, name)

            sparsity = self.computeSparsity(model_i)
            self.models.append((model_i, sparsity))

    @override
    def plotMetric(self, metrics, metric_name):
        """ Plot the results from evaluation and the CKA 

        Args:
            accuracies (list): List of accuracies (must align with pruned model sparsitites)
        """

        # Plot Accuracy Results
        plt.figure()
        plt.plot(self.strengths, metrics, label=f"Unstructured Pruning: {metric_name}")
        plt.title("Validation Accuracy vs Encoder Pruning Percentage")
        plt.xlabel("Encoder Layer's Sparsity")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "accuracy_plot.png"))





