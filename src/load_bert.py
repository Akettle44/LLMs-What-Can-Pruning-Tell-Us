import os
import torch
from transformers import BertConfig, BertTokenizer, BertModel
from .model import BertCustomHead
  
def load_config_and_tokenizer(model_dir):
    """
    Load the BERT configuration and tokenizer from a specified directory.
    """
    config = BertConfig.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir, useFast=True)
    return config, tokenizer

def load_model(model_dir, num_classes, task_type):
    """
    Load a fine-tuned BERT model from a specified directory.
    """
    config, tokenizer = load_config_and_tokenizer(model_dir)
    model = BertCustomHead(config, num_classes, task_type)
    state_dict_path = os.path.join(model_dir, 'pytorch_model.bin')
    # Map to CPU by default, user converts to CUDA in their script
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def load_all(model_dir, num_classes, task_type):
    """
    Load the model, configuration, and tokenizer from a specified directory.
    """
    config, tokenizer = load_config_and_tokenizer(model_dir)
    model = load_model(model_dir, num_classes, task_type)
    return model, config, tokenizer

