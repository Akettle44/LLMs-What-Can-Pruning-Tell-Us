import os
import torch
from transformers import BertConfig, BertTokenizer
from .model import BertCustom

def save_model(model: BertCustom, root_dir, model_name):
    """ Save the bert model to disk (using huggingface)

    Args:
        model (pytorch nn.module): Model
        model_dir (str): Directory to save model to
    """
    task_dir = os.path.join(os.path.join(root_dir, "models"), model.task_type)
    model_dir = os.path.join(task_dir, model_name)
    if( not os.path.exists(task_dir)):
        os.mkdir(task_dir)
    
    model.save_pretrained(model_dir)

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
    model = BertCustom(config, num_classes, task_type)
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

