import torch
from typing import List
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

    def changeLossFunction(self, newLossFunction):
        """ This function provides the ability to change a HeadModule's loss function.
        """
        if not issubclass(torch.nn):
            raise ValueError("HeadModule::changeLossFunction: invalid loss function provided")

        self.loss_function = newLossFunction

class BertCustom(torch.nn.Module):
    def __init__(self, config, num_classes, tokenizer, task_type, use_pretrained=True, 
                 default_model='bert-base-uncased', override_loss_function=None):
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
            self.config = BertConfig(hidden_dropout_prob=0.4, attention_probs_dropout_prob=0.4)
            self.model = BertModel(self.config)
        elif self.config is None:
            self.model = BertModel.from_pretrained(default_model, hidden_dropout_prob=0.4, attention_probs_dropout_prob=0.4)
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

        # Override loss function if user provided
        if override_loss_function is not None:
            self.task_modules[self.task_type].changeLossFunction(override_loss_function)

        # Hanlde unsupported task
        if task_type not in list(self.task_modules.keys()):
            raise ValueError(f'Invalid task type. Supported types: {list(self.task_modules.keys())}') 

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, label=None):

            #'summarization': _summary_
        #    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #    intermediate_output = outputs.last_hidden_state
        #    attentions = outputs.attentions

        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        intermediate_output = outputs.pooler_output
        attentions = outputs.attentions

        # Compute loss as well
        if(label is not None):
            logits, loss = self.task_modules[self.task_type](intermediate_output, label)
        else:
            logits = self.task_modules[self.task_type](intermediate_output)
            loss = None

        return logits, attentions, intermediate_output, loss

    def freezeLayers(self, freeze_idxs: List[int]):
        """ Freeze particular layers during training
            Typically used to mitigate overfitting
        Args:
            freeze_idxs (list): List of pytorch encoder layers to freeze
        """
        # Freeze ecnoder layers
        for idx in freeze_idxs:
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = False

    # TODO: Eventually combine this into forward pass, should have all necessary information
    def computeMetric(self, outputs, targets):
        """ Compute the metric associated with a particular task

        Args:
            outputs (tuple): Outputs from forward pass
            targets (_type_): _description_
        """

        # Vanilla unweighted accuracy
        if self.task_type == 'sequence_classification':
            logits = outputs[0]
            probabilities = torch.argmax(torch.sigmoid(logits), axis=1)
            correct = torch.count_nonzero(probabilities == targets)
            if len(targets) == 0:
                return 0

            return correct.item() / len(targets)
            