import os
import torch
from transformers import BertConfig, BertTokenizer, BertModel

class HeadModule(torch.nn.Module):
    """ Store the final output layer and loss function for a particular task together.
    """

    def __init__(self, output_layer, loss_function):
        super(HeadModule, self).__init__()
        self.output_layer = output_layer
        self.loss_function = loss_function

    def forward(self, x, target=None):
        """ Forward pass through final layer and score calculation. 
            We compute the loss function within the forward pass because
            the loss function is task specific.

        Args:
            x (torch.tensor): Final layer input tensor
            target (torch.tensor, optional): Groundtruth
        """

        out = self.output_layer(x)
        if target is not None:
            loss = self.loss_function(out, target)
            return out, loss
        return out

class BertCustomHead(torch.nn.Module):
    def __init__(self, config, num_classes, task_type):
        super(BertCustomHead, self).__init__()

        # Handle empty config
        if config is None:
            config = BertConfig()
            print("Config is None; Defaulting to HuggingFace BertConfig")
        else:
            self.bert = BertModel(config)

        # Task Options
        self.task_modules = torch.nn.ModuleDict({
            'sequence_classification': 
            HeadModule(torch.nn.Linear(config.hidden_size, num_classes), torch.nn.CrossEntropyLoss()),
            'token_classification':
            HeadModule(torch.nn.Linear(config.hidden_size, num_classes), torch.nn.CrossEntropyLoss()),
            'multiple_choice': 
            HeadModule(torch.nn.Linear(config.hidden_size, 1), torch.nn.BCEWithLogitsLoss())
        })
            #'summarization': 
            #HeadModule(torch.nn.Linear(config.hidden_size, config.vocab_size), torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id))
                
        # Hanlde unsupported task
        if task_type not in list(self.task_modules.keys()):
            raise ValueError(f'Invalid task type. Supported types: {list(self.task_modules.keys())}') 

        self.task_type = task_type

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, decoder_input_ids=None, target=None):

        # TODO: Add support for summarization
        #if self.task_type == 'summarization':
        #    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #    intermediate_output = outputs.last_hidden_state
        #    attentions = outputs.attentions

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        intermediate_output = outputs.pooler_output
        attentions = outputs.attentions

        # Compute loss as well
        if(target is not None):
            logits, loss = self.task_modules[self.task_type](intermediate_output, target)
        else:
            logits = self.task_modules[self.task_type](intermediate_output)
            loss = None

        return logits, attentions, intermediate_output, loss
   
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

