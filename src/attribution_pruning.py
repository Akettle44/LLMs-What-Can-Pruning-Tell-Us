import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
import torch.nn as nn
import os
import torch.nn.utils.prune as prune
from datasets import load_dataset, load_metric
import json
import os
import time

model_dir = '/pub/gautamb1/cs295/checkpoints/bert_finetuned_sst2'
start = time.time()

class BertCustomHead(nn.Module):
    def __init__(self, config, num_classes, task_type='sequence_classification'):
        super(BertCustomHead, self).__init__()
        self.bert = BertModel(config)
        self.task_type = task_type

        self.heads = nn.ModuleDict({
            'sequence_classification': nn.Linear(config.hidden_size, num_classes),
            'token_classification': nn.Linear(config.hidden_size, num_classes),
            'multiple_choice': nn.Linear(config.hidden_size, 1),
            'summarization': nn.Linear(config.hidden_size, config.vocab_size)
        })

        self.loss_fns = {
            'sequence_classification': nn.CrossEntropyLoss(),
            'token_classification': nn.CrossEntropyLoss(),
            'multiple_choice': nn.BCEWithLogitsLoss(),
            'summarization': nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        }

        if task_type not in self.heads:
            raise ValueError("Invalid task type. Supported types: 'sequence_classification', 'token_classification', 'multiple_choice', 'summarization'")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, decoder_input_ids=None):
        if self.task_type == 'summarization':
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            logits = self.heads['summarization'](sequence_output)
            return logits
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            return self.heads[self.task_type](pooled_output)

config = BertConfig.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

task = "sst2"
batch_size = 16
num_classes = 2  # SST-2 has binary labels: positive and negative

model = BertCustomHead(config, num_classes, task_type='sequence_classification')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict_path = os.path.join(model_dir, 'pytorch_model.bin')
state_dict = torch.load(state_dict_path, map_location=device) #Dont map it to CPU if ur using a GPU!!!
model.load_state_dict(state_dict)
model.to(device)

# Load dataset and tokenizer
dataset = load_dataset("glue", task)
metric = load_metric("glue", task, trust_remote_code=True)  # Use evaluate library in future

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

# Tokenize the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Prepare DataLoader
SST_train_dataset = encoded_dataset['train']
SST_validation_dataset = encoded_dataset['validation']
SST_train_dataloader = DataLoader(SST_train_dataset, shuffle=True, batch_size=batch_size)
SST_validation_dataloader = DataLoader(SST_validation_dataset, batch_size=batch_size)
print(SST_train_dataloader.dataset)
# Define model
class BertCustomHead(nn.Module):
    def __init__(self, config, num_classes, task_type='sequence_classification'):
        super(BertCustomHead, self).__init__()
        self.bert = BertModel(config)
        self.task_type = task_type

        self.heads = nn.ModuleDict({
            'sequence_classification': nn.Linear(config.hidden_size, num_classes),
            'token_classification': nn.Linear(config.hidden_size, num_classes),
            'multiple_choice': nn.Linear(config.hidden_size, 1),
            'summarization': nn.Linear(config.hidden_size, config.vocab_size)
        })

        self.loss_fns = {
            'sequence_classification': nn.CrossEntropyLoss(),
            'token_classification': nn.CrossEntropyLoss(),
            'multiple_choice': nn.BCEWithLogitsLoss(),
            'summarization': nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        }

        if task_type not in self.heads:
            raise ValueError("Invalid task type. Supported types: 'sequence_classification', 'token_classification', 'multiple_choice', 'summarization'")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, decoder_input_ids=None):
        if self.task_type == 'summarization':
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            logits = self.heads['summarization'](sequence_output)
            return logits
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            return self.heads[self.task_type](pooled_output)

# Hook to capture hidden states and gradients
hidden_states = [None] * model.bert.config.num_hidden_layers
hidden_states_grad = [None] * model.bert.config.num_hidden_layers

def forward_hook(layer_idx):
    def hook(module, input, output):
        hidden_states[layer_idx] = output
    return hook

def backward_hook(layer_idx):
    def hook(module, grad_input, grad_output):
        hidden_states_grad[layer_idx] = grad_output[0]
    return hook

def compute_attributions_for_all_layers(model, inputs, attention_mask, labels):
    global hidden_states, hidden_states_grad

    # Register hooks for all layers
    hooks = []
    for layer_idx in range(model.bert.config.num_hidden_layers):
        hooks.append(model.bert.encoder.layer[layer_idx].output.register_forward_hook(forward_hook(layer_idx)))
        hooks.append(model.bert.encoder.layer[layer_idx].output.register_backward_hook(backward_hook(layer_idx)))

    # Forward pass
    logits = model(input_ids=inputs, attention_mask=attention_mask)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward()
    
    # Ensure hooks have captured hidden states and gradients
    for layer_idx in range(model.bert.config.num_hidden_layers):
        assert hidden_states[layer_idx] is not None, f"Hidden states not captured for layer {layer_idx}"
        assert hidden_states_grad[layer_idx] is not None, f"Hidden states gradients not captured for layer {layer_idx}"

    attributions_all_layers = []

    # Compute attributions for each layer
    for layer_idx in range(model.bert.config.num_hidden_layers):
        hidden_states_detached = hidden_states[layer_idx].detach()
        hidden_states_grad_detached = hidden_states_grad[layer_idx].detach()

        # Compute attributions for neurons in this layer
        attributions = hidden_states_grad_detached * hidden_states_detached

        # Sum attributions across the batch dimension
        attributions_sum = attributions.sum(dim=0).sum(dim=0)
        attributions_all_layers.append(attributions_sum.cpu())

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    return attributions_all_layers


def derive_task_specific_neuron_indices(model, data_loader, pruning_rates):
    num_layers = model.bert.config.num_hidden_layers
    num_neurons_per_layer = model.bert.config.hidden_size
    neuron_indices_to_keep = {}

    attribution_scores_all_layers = [torch.zeros(num_neurons_per_layer) for _ in range(num_layers)]


    for pruning_rate in pruning_rates:
            neuron_indices_to_keep[pruning_rate] = {}

    for batch in data_loader:
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        model.zero_grad()

        attributions_all_layers = compute_attributions_for_all_layers(model, inputs, attention_mask, labels)

        # Aggregate attribution scores for each layer
        for layer_idx in range(num_layers):
            attribution_scores_all_layers[layer_idx] += attributions_all_layers[layer_idx]

    for layer_idx in range(num_layers):
        # Average the attribution scores
        attribution_scores_all_layers[layer_idx] /= len(data_loader)

        # Get the top (1 - pruning_rate) neurons
        for pruning_rate in pruning_rates:
            num_neurons_to_keep = int((1 - pruning_rate) * num_neurons_per_layer)
            _, top_indices = torch.topk(torch.abs(attribution_scores_all_layers[layer_idx]), num_neurons_to_keep)
            neuron_indices_to_keep[pruning_rate][layer_idx] = top_indices.tolist()
        
    return neuron_indices_to_keep

def custom_pruning_method(layer, neuron_indices, num_neurons_per_layer):
    mask = torch.zeros(num_neurons_per_layer, dtype=torch.float32)
    mask[neuron_indices] = 1
    return mask


def prune_model(model, neuron_indices_to_keep, pruning_rate):
    num_neurons_per_layer = model.bert.config.hidden_size

    for layer_idx, neuron_indices in neuron_indices_to_keep[pruning_rate].items():
        layer = model.bert.encoder.layer[layer_idx]
        layer.to('cpu')

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
        prune.custom_from_mask(layer.attention.output.LayerNorm, name='weight', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.custom_from_mask(layer.attention.output.LayerNorm, name='bias', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.remove(layer.attention.output.dense, name='weight')
        prune.remove(layer.attention.output.dense, name='bias')
        prune.remove(layer.attention.output.LayerNorm, name='weight')
        prune.remove(layer.attention.output.LayerNorm, name='bias')


        # Prune the intermediate dense layer
        mask = custom_pruning_method(layer.intermediate.dense, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.intermediate.dense, name='weight', mask=mask.unsqueeze(0).expand_as(layer.intermediate.dense.weight))
        prune.remove(layer.intermediate.dense, name='weight')

        # Prune the output dense layer
        mask = custom_pruning_method(layer.output.dense, neuron_indices, num_neurons_per_layer)
        prune.custom_from_mask(layer.output.dense, name='weight', mask=mask.unsqueeze(1).expand_as(layer.output.dense.weight))
        prune.custom_from_mask(layer.output.dense, name='bias', mask=mask.expand_as(layer.output.dense.bias))
        prune.custom_from_mask(layer.output.LayerNorm, name='weight', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.custom_from_mask(layer.output.LayerNorm, name='bias', mask=mask.expand_as(layer.attention.output.dense.bias))
        prune.remove(layer.output.dense, name='weight')
        prune.remove(layer.output.dense, name='bias')
        prune.remove(layer.output.LayerNorm, name='weight')
        prune.remove(layer.output.LayerNorm, name='bias')

def calcSparsity(model):
    zeros = 0
    elements = 0

    for _, param in model.named_parameters():

        zero_count = torch.sum(param==0.0).item()
        zeros += zero_count

        element_count = param.numel()
        elements += element_count

    if(elements == 0):
        return 0

    return zeros / elements

pruning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Check sparsity
print(calcSparsity(model))

# Derive task-specific neuron indices
neuron_indices_to_keep = derive_task_specific_neuron_indices(model, SST_train_dataloader, pruning_rates)

import pickle

with open('sst2_attributions.pickle', 'wb') as handle:
    pickle.dump(neuron_indices_to_keep, handle, protocol=pickle.HIGHEST_PROTOCOL)


for pruning_rate in pruning_rates:
    # Prune model
    prune_model(model, neuron_indices_to_keep, pruning_rate)
    print("Model pruned.")
    
    # Check sparsity
    print(calcSparsity(model))


    # Save Model
    save_directory = "/pub/gautamb1/cs295/checkpoints/bert_prune_sst2_" + str(pruning_rate)
    os.makedirs(save_directory, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
    tokenizer.save_pretrained(save_directory)
    model_config_dict = model.bert.config.to_dict()
    model_config_dict.update({
        "architectures": ["BertCustomHead"],
        "num_labels": 2,  
        "task_specific_params": {
            "sequence_classification": {
                "num_classes": 2  
            },
            "token_classification": {
                "num_classes": 2  
            },
            "multiple_choice": {
                "num_classes": 4  
            },
            "summarization": {
                "vocab_size": model_config_dict["vocab_size"]  
            }
        }
    })
    config_path = os.path.join(save_directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config_dict, f, indent=2)
    print(f"Configuration saved to {config_path}")
    print(time.time()-start)