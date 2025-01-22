import os
import torch
from transformers import BertConfig, BertTokenizer
from .model import BertCustom
from utils.utils import Utils

# TODO: Push hyperparemeter config file here as well
def saveModelToDisk(model: BertCustom, root_dir, model_name):
    """ Save the bert model to disk (using huggingface)

    Args:
        model (pytorch nn.module): Model
        model_dir (str): Directory to save model to
    """

    task_dir = os.path.join(os.path.join(root_dir, "models"), model.task_type)
    if( not os.path.exists(task_dir)):
        os.mkdir(task_dir)

    save_dir = os.path.join(task_dir, model_name)
    if ( not os.path.exists(save_dir)):
        os.mkdir(save_dir)
    model_dir = os.path.join(save_dir, "model.pth")

    # Write the config, model, and tokenizer to disk
    model.tokenizer.save_pretrained(save_dir)
    model.config.save_pretrained(save_dir)    
    torch.save(model.state_dict(), model_dir)

    # Write model metadata to disk
    metadata = os.path.join(save_dir, "metadata.txt")
    with open(metadata, 'w') as f:
        f.write('Legend: num_classes, task_type \n')
        f.write(f"{model.num_classes}, {model.task_type}")

# TODO: Adjust bert constructor to take in dropout arguments so that 
# here and main can have access to it
def loadModelFromDisk(model_dir):
    """ Load the config, tokenizer, and model from disk (local)
    If you want to use a pretrained model directly from HuggingFace, 
    you can override the default model parameter when creating a 
    BertCustom model

    Args:
        model_dir (str): path to model on disk
    """

    # Load Hyperparameters
    task_name = model_dir.split('/')[-1].split('-')[0]
    hyp_dir = os.path.dirname(os.path.dirname(model_dir)) # Insanely ratchet
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(hyp_dir, 'hyps'), task_name + '.txt'))

    config = BertConfig.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir, useFast=True)

    metadata = os.path.join(model_dir, "metadata.txt")
    with open(metadata, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 1:
                num_classes, task_type = line.split(',')
                num_classes = int(num_classes)
                task_type = task_type.strip()

    bert = BertCustom(config, num_classes, tokenizer, task_type)
    bert.freezeLayers(hyps['freezeLayers']) # TODO: Check to make sure layers need to be frozen

    # TODO: Improve this, should be internal to class
    state_dict = torch.load(os.path.join(model_dir, 'model.pth'))
    bert.load_state_dict(state_dict)

    return bert, hyps
