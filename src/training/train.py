### This file performs training in PyTorch

import torch
from model.model import BertCustom

class PtTrainer():

    def __init__(self, model, dataset, optimizer=None):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyps = {}

        if self.optimizer is None:
            self.setDefaultOptimizer()

    def setDefaultOptimizer(self):
        """ Select appropriate opitimizer and associated params
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

    def setDevice(self, device):
        """ Updates the device  
        """
        self.device = device

    def sendToDevice(self):
        """ Places objects on correct device prior to training
        """
        self.model.to(self.device)

    def setHyps(self, **hyps):
        """ Grab all hyperparameters and their associated value
        """
        for key, value in hyps.items():
            self.hyps[key] = value

    def fineTune(self):
        
        # Unpack dataset
        train_loader = self.dataset.train_loader
        val_loader = self.dataset.val_loader

        train_loss = []
        train_accs = []
        val_loss = []
        val_accs = []

        for epoch in range(self.hyps['epochs']):

            ### TRAIN SINGLE EPOCH ###
            # Set training mode in PyTorch
            self.model.train() 
            train_epoch_loss = []
            train_batch_accs = []
            
            # One iteration over dataset
            for batch in train_loader:
                
                # Zero out accumulation
                self.optimizer.zero_grad()

                # Generate prediction
                batch = {k: v.to(self.device) for k,v in batch.items()}
                outputs = self.model(**batch)

                print(outputs)

                # Compute outputs + loss; update parameters
                loss = outputs[-1]
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())

                # Compute Accuracy
                logits = outputs[0]
                probabilities = torch.argmax(torch.sigmoid(logits), axis=1)
                correct = torch.count_nonzero(probabilities == batch['label'])
                train_batch_accs.append(correct / batch['label'].shape[0])
            
            # Average loss and accuracy over epoch
            train_loss.append(torch.mean(torch.tensor(train_epoch_loss)))
            train_accs.append(torch.mean(torch.tensor(train_batch_accs)))
            ### END SINGLE EPOCH TRAIN ###

            ### EPOCH VALIDATION ###
            # Set eval mode
            self.model.eval() 
            val_epoch_loss = []
            val_batch_accs = []

            with torch.no_grad():
                # Perform validation
                for batch in val_loader:
                    
                    # Run inference
                    batch = {k: v.to(self.device) for k,v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs[-1]
                    val_epoch_loss.append(loss.item())

                    # Compute Accuracy
                    logits = outputs[0]
                    probabilities = torch.argmax(torch.sigmoid(logits), axis=1)
                    correct = torch.count_nonzero(probabilities == batch['label'])
                    val_batch_accs.append(correct / batch['label'].shape[0])
            
            val_loss.append(torch.mean(torch.tensor(val_epoch_loss)))
            val_accs.append(torch.mean(torch.tensor(val_batch_accs)))
            ### END EPOCH VALIDATION ###

        return train_loss, train_accs, val_loss, val_accs