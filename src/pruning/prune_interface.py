# Abstract Class for different pruning implementations to follow

import torch
import torch.nn.utils.prune as prune
import transformers
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from torch_cka import CKA
from abc import ABC, abstractmethod

class Pruner(ABC):
    @abstractmethod
    def __init__(self, baseline, save_dir, layers, strengths):
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
        self.save_dir = save_dir

    @abstractmethod
    def prune(self):
        pass

    def compareModels(self, dataloader, device):
        """ Compare models using CKA. TODO: Eventually break this out into its own
            class so that models can be compared in different ways

        Returns:
            _type_: _description_
        """

        # Isolate CKA comparison to BERT encoder only
        model1_names = [f"model.encoder.layer.{i}.attention" for i in range(0, 12)]
        model2_names = [s for s in model1_names] # Perform Copy

        # Be careful to watch VRAM when running this
        accuracies = [] # Measured on dataloader passed to this function
        for i in range(1, self.num_models):
            with torch.no_grad():
                # Name models for plot
                model1_name = "Baseline"
                model2_name = f"{self.baseline.task_type}-Pruned-{round(self.strengths[i], 2)}"

                # Select model, Always compare against baseline
                model1, sparsity1 = self.models[0] # Baseline
                model2, sparsity2 = self.models[i] # Pruned Model
                model1 = model1.to(device)
                model2 = model2.to(device)

                if(i == 1):
                    # Compute accuracy for baseline model on first pass
                    accuracies.append(Pruner.performEval(model1, dataloader, device))

                accuracies.append(Pruner.performEval(model2, dataloader, device))

                # Compute CKA
                cka = CKA(model1, model2, model1_name=model1_name, model2_name=model2_name, model1_layers=model1_names, model2_layers=model2_names, device=device)
                cka.compare(dataloader) 
                results = cka.export()
                #print(results)

                self.plotCka(cka, model1_name, model2_name)

                # Remove model from GPU mem
                del model2 # Reference therefore list idx deleted as well
                torch.cuda.empty_cache()

        self.plotMetric(accuracies, "Accuracy")

        return accuracies

    def plotCka(self, cka, model1_name, model2_name):
        """ Plot the CKA results

        Args:
            cka (cka): cka object
            model1_name (str): model1 name
            model2_name (str): model2 name
        """
        # Plot CKA Results
        plot_name = os.path.join(self.save_dir, model2_name + ".png")
        cka.plot_results(plot_name, f"{model1_name} vs {model2_name}")

    @abstractmethod
    def plotMetric(self, metrics, metric_name):
        pass

    @staticmethod
    def performEval(model, dataloader, device):
        """ Perform evaluation of the model

        Args:
            model (torch.nn.module)
            dataloader (torch.dataloader)
        Returns:
            Metric score for particular task
        """

        for batch in tqdm(dataloader):        
            # Run inference
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            met_score = model.computeMetric(outputs, batch['label'])

        return met_score

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
