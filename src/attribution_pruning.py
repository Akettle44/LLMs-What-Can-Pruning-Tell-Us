'''
File for Attribution-based Pruning
Tested for cola task
'''

import torch
from transformers import BertTokenizer, BertConfig, BertModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.utils.prune as prune
import time
import numpy as np
import hf_train as local_trainer
import prune as prune_utils
import os

class BertCustomHead(nn.Module):
    """
    Defines the BertCustomHead module. 
    """
    def __init__(self, config, num_classes, task_type='sequence_classification'):
        super(BertCustomHead, self).__init__()
        self.bert = BertModel(config)
        self.task_type = task_type
        self.heads = {
            'sequence_classification': nn.Linear(config.hidden_size, num_classes),
            'token_classification': nn.Linear(config.hidden_size, num_classes),
            'multiple_choice': nn.Linear(config.hidden_size, 1)
        }
        self.loss_fns = {
            'sequence_classification': nn.CrossEntropyLoss(),
            'token_classification': nn.CrossEntropyLoss(),
            'multiple_choice': nn.CrossEntropyLoss()
        }
        
        if task_type not in self.heads:
            raise ValueError("Invalid task type. Supported types: 'sequence_classification', 'token_classification', 'multiple_choice'")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, next_sentence_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        return self.heads[self.task_type](pooled_output)

def forward_hook(module, input, output):
    global hidden_states
    hidden_states = output

def backward_hook(module, grad_input, grad_output):
    global hidden_states_grad
    hidden_states_grad = grad_output[0]

def compute_attributions_for_layer(model, layer_idx, inputs, attention_mask, labels):
    global hidden_states, hidden_states_grad

    # Register hooks
    hook_forward = model.bert.encoder.layer[layer_idx].output.register_forward_hook(forward_hook)
    hook_backward = model.bert.encoder.layer[layer_idx].output.register_backward_hook(backward_hook)

    # Forward pass
    logits = model(input_ids=inputs, attention_mask=attention_mask)

    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward()
    
    # Ensure hooks have captured hidden states and gradients
    assert hidden_states is not None, "Hidden states not captured"
    assert hidden_states_grad is not None, "Hidden states gradients not captured"

    # Detach outputs and gradients
    hidden_states_detached = hidden_states.detach()
    hidden_states_grad_detached = hidden_states_grad.detach()

    # Compute attributions for neurons
    attributions = hidden_states_grad_detached * hidden_states_detached

    # Sum attributions across the batch dimension
    attributions_sum = attributions.sum(dim=0).sum(dim=0)

    # Clean up hooks
    hook_forward.remove()
    hook_backward.remove()

    return attributions_sum.cpu()

def derive_task_specific_neuron_indices(model, data_loader, pruning_rate):
    num_layers = model.bert.config.num_hidden_layers
    num_neurons_per_layer = model.bert.config.hidden_size
    neuron_indices_to_keep = {}

    for layer_idx in range(num_layers):
        attribution_scores = torch.zeros(num_neurons_per_layer)

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            model.zero_grad()

            attributions = compute_attributions_for_layer(model, layer_idx, inputs, attention_mask, labels)
            attribution_scores += attributions

        # Average the attribution scores
        attribution_scores /= len(data_loader)

        # Get the top (1 - pruning_rate) neurons
        num_neurons_to_keep = int((1 - pruning_rate) * num_neurons_per_layer)
        _, top_indices = torch.topk(attribution_scores, num_neurons_to_keep)
        neuron_indices_to_keep[layer_idx] = top_indices.tolist()

    return neuron_indices_to_keep

def custom_pruning_method(layer, neuron_indices, num_neurons_per_layer):
    mask = torch.zeros(num_neurons_per_layer, dtype=torch.float32)
    mask[neuron_indices] = 1
    return mask


def prune_model(model, neuron_indices_to_keep):
    num_neurons_per_layer = model.bert.config.hidden_size

    for layer_idx, neuron_indices in neuron_indices_to_keep.items():
        layer = model.bert.encoder.layer[layer_idx]
        # Prune the attention self query layer
        mask = custom_pruning_method(layer.attention.self.query, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.attention.self.query, name='weight', mask=mask.unsqueeze(0).expand_as(layer.attention.output.dense.weight))
        prune.custom_from_mask(layer.attention.self.query, name='bias', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.remove(layer.attention.self.query, name='weight')
        prune.remove(layer.attention.self.query, name='bias')

        # Prune the attention self key layer
        mask = custom_pruning_method(layer.attention.self.key, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.attention.self.key, name='weight', mask=mask.unsqueeze(0).expand_as(layer.attention.output.dense.weight))
        prune.custom_from_mask(layer.attention.self.key, name='bias', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.remove(layer.attention.self.key, name='weight')
        prune.remove(layer.attention.self.key, name='bias')

        # Prune the attention self value layer
        mask = custom_pruning_method(layer.attention.self.value, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.attention.self.value, name='weight', mask=mask.unsqueeze(0).expand_as(layer.attention.output.dense.weight))
        prune.custom_from_mask(layer.attention.self.value, name='bias', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.remove(layer.attention.self.value, name='weight')
        prune.remove(layer.attention.self.value, name='bias')


        # Prune the attention output dense layer
        mask = custom_pruning_method(layer.attention.output.dense, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.attention.output.dense, name='weight', mask=mask.unsqueeze(0).expand_as(layer.attention.output.dense.weight))
        prune.custom_from_mask(layer.attention.output.dense, name='bias', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.remove(layer.attention.output.dense, name='weight')
        prune.remove(layer.attention.output.dense, name='bias')


        # Prune the intermediate dense layer
        mask = custom_pruning_method(layer.intermediate.dense, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.intermediate.dense, name='weight', mask=mask.unsqueeze(0).expand_as(layer.intermediate.dense.weight))
        prune.remove(layer.intermediate.dense, name='weight')

        # Prune the output dense layer
        mask = custom_pruning_method(layer.output.dense, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.output.dense, name='weight', mask=mask.unsqueeze(1).expand_as(layer.output.dense.weight))
        prune.custom_from_mask(layer.output.dense, name='bias', mask=mask.expand_as(layer.output.dense.bias))
        prune.remove(layer.output.dense, name='weight')
        prune.remove(layer.output.dense, name='bias')



def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)


def main():
    global tokenizer
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "models")
    # From HF
    checkpoint = "bert-base-uncased"

    # Constants
    task = "cola"
    num_labels = 2
    batch_size = 16
    learning_rate = 2e-5
    epochs = 5
    weight_decay = 0.01
    metric_name = "matthews_correlation"

    # Grab dataset
    dataset = local_trainer.chooseTaskHF(task)
    train_dataset = load_dataset('glue', 'cola', split='train[:20]')
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(tokenized_train_dataset, batch_size=8)
    print(train_loader.dataset)

    # Grab model
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertCustomHead(config, num_classes=2, task_type='sequence_classification')

    # Hook to capture hidden states and gradients
    hidden_states = None
    hidden_states_grad = None

    # Evaluate the pruned model on the validation dataset
    print(prune_utils.calcSparsity(model))
    
    # Pruning rate
    pruning_rate = 0.5
    # Derive task-specific neuron indices
    neuron_indices_to_keep = derive_task_specific_neuron_indices(model, train_loader, pruning_rate)

    # Prune model
    prune_model(model, neuron_indices_to_keep)
    print("Model pruned.")


    # Evaluate the pruned model on the validation dataset
    print(prune_utils.calcSparsity(model))



if __name__ == '__main__':
    main()