import os
import torch
from transformers import BertConfig, BertTokenizer, BertModel
import torch.nn as nn


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


def load_config_and_tokenizer(model_dir):
    """
    Load the BERT configuration and tokenizer from a specified directory.
    """
    config = BertConfig.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return config, tokenizer


def load_model(model_dir, num_classes, task_type='sequence_classification'):
    """
    Load a fine-tuned BERT model from a specified directory.
    """
    config, _ = load_config_and_tokenizer(model_dir)
    model = BertCustomHead(config, num_classes, task_type)
    state_dict_path = os.path.join(model_dir, 'pytorch_model.bin')
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))  # Map to 'cpu' or 'cuda'
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_all(model_dir, num_classes, task_type='sequence_classification'):
    """
    Load the model, configuration, and tokenizer from a specified directory.
    """
    config, tokenizer = load_config_and_tokenizer(model_dir)
    model = load_model(model_dir, num_classes, task_type)
    return model, config, tokenizer

