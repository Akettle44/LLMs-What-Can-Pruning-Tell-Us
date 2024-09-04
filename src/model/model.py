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

class BertCustom(torch.nn.Module):
    def __init__(self, config, num_classes, tokenizer, task_type, use_pretrained=True, 
                 default_model='bert-base-uncased'):
        super(BertCustom, self).__init__()

        self.config = config
        self.num_classes = num_classes
        self.task_type = task_type
        self.pretrained = use_pretrained
        self.tokenizer = tokenizer

        # Tokenizer
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(default_model)

        # Config and model
        if self.config is None and not use_pretrained:
            self.config = BertConfig()
            self.model = BertModel(self.config)
        elif self.config is None:
            self.model = BertModel.from_pretrained(default_model)
            self.config = self.model.config
        else:
            self.model = BertModel(self.config)
        
        # Task Options
        self.task_modules = torch.nn.ModuleDict({
            'sequence_classification': 
            HeadModule(torch.nn.Linear(self.config.hidden_size, num_classes), torch.nn.CrossEntropyLoss()),
            'token_classification':
            HeadModule(torch.nn.Linear(self.config.hidden_size, num_classes), torch.nn.CrossEntropyLoss()),
            'multiple_choice':  
            HeadModule(torch.nn.Linear(self.config.hidden_size, 1), torch.nn.BCEWithLogitsLoss())
        })

            #'summarization': 
            #HeadModule(torch.nn.Linear(config.hidden_size, config.vocab_size), torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id))
                        
        # Hanlde unsupported task
        if task_type not in list(self.task_modules.keys()):
            raise ValueError(f'Invalid task type. Supported types: {list(self.task_modules.keys())}') 

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, decoder_input_ids=None, target=None):

            #'summarization': _summary_
        #    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #    intermediate_output = outputs.last_hidden_state
        #    attentions = outputs.attentions

        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        intermediate_output = outputs.pooler_output
        attentions = outputs.attentions

        # Compute loss as well
        if(target is not None):
            logits, loss = self.task_modules[self.task_type](intermediate_output, target)
        else:
            logits = self.task_modules[self.task_type](intermediate_output)
            loss = None

        return logits, attentions, intermediate_output, loss