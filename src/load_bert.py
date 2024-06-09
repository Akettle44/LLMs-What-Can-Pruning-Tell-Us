import os
import torch
from transformers import BertConfig, BertTokenizer, BertModel

class BertCustomHead(torch.nn.Module):
    def __init__(self, config, num_classes, tokenizer, task_type='sequence_classification'):
        super(BertCustomHead, self).__init__()
        self.bert = BertModel(config)
        self.task_type = task_type

        self.heads = torch.nn.ModuleDict({
            'sequence_classification': torch.nn.Linear(config.hidden_size, num_classes),
            'token_classification': torch.nn.Linear(config.hidden_size, num_classes),
            'multiple_choice': torch.nn.Linear(config.hidden_size, 1),
            'summarization': torch.nn.Linear(config.hidden_size, config.vocab_size)
        })

        self.loss_fns = {
            'sequence_classification': torch.nn.CrossEntropyLoss(),
            'token_classification': torch.nn.CrossEntropyLoss(),
            'multiple_choice': torch.nn.BCEWithLogitsLoss(),
            'summarization': torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        }

        if task_type not in self.heads:
            raise ValueError("Invalid task type. Supported types: 'sequence_classification', 'token_classification', 'multiple_choice', 'summarization'")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, decoder_input_ids=None):

        if self.task_type == 'summarization':
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            task_output = outputs.last_hidden_state
            attentions = outputs.attentions
            logits = self.heads[self.task_type](task_output)
            
        elif self.task_type == 'sequence_classification':
            outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            attentions = outputs.attentions
            task_output = outputs.last_hidden_state
            logits = self.heads[self.task_type](task_output)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            task_output = outputs.pooler_output
            attentions = outputs.attentions
            return self.heads[self.task_type](task_output)

        return task_output, attentions, logits

def load_config_and_tokenizer(model_dir):
    """
    Load the BERT configuration and tokenizer from a specified directory.
    """
    config = BertConfig.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir, useFast=True)
    return config, tokenizer


def load_model(model_dir, num_classes, task_type='sequence_classification'):
    """
    Load a fine-tuned BERT model from a specified directory.
    """
    config, tokenizer = load_config_and_tokenizer(model_dir)
    model = BertCustomHead(config, num_classes, tokenizer, task_type)
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

