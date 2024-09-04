### This file performs training in PyTorch

import torch
from ..model.model import BertCustom

class PtTrainer():

    def __init__(self, model, dataset, optimizer):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyps = {}

    def setDefaultOptimizer(self):
        """ Select appropriate opitimizer and associated params
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

    def setDevice(self):
        """ Places objects on correct device prior to training
        """
        self.model.to(self.device)
        self.optimizer.to(self.device)

    def setHyps(self, **hyps):
        """ Grab all hyperparameters and their associated value
        """
        for key, value in hyps.items():
            self.hyps[key] = value

    def fineTune(self):
        
        # Unpack hyperparameters
        # TODO

        train_loss = []
        val_loss = []

        for epoch in range(self.hyps['epochs']):

            ### TRAINING ###
            self.model.train() # Set training mode in PyTorch
            epoch_train_loss = []
            
            # One iteration over dataset
            for batch in self.dataset.train_loader:
                batch = tuple(sample.to(self.device) for sample in batch)

                # Best way to unpack ?
                inputs, masks, labels = batch


                self.optimizer.zero_grad()
                
                outputs = model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()

                epoch_train_loss.append(loss.item())
            
            # Average loss over epoch
            train_loss.append(torch.mean(torch.tensor(epoch_train_loss)))

            ### END TRAINING ###

            ### VALIDATION ###
            epoch_val_loss = []
            
            model.eval() # Set eval mode
            val_accuracy = 0

            with torch.no_grad():
                # Perform validation
                for batch in val_loader:
                    batch = tuple(t.to(device) for t in batch)
                    inputs, masks, labels = batch

                    outputs = model(inputs, attention_mask=masks)
                    # Grab raw logits
                    logits = outputs.logits
                    logits = logits.detach().cpu().numpy()
                    label_ids = labels.to('cpu').numpy()
                    print(logits)


                    val_accuracy += (logits.argmax(axis=1) == label_ids).mean().item()
            
            #avg_val_accuracy = val_accuracy / len(val_loader)
            #print("Epoch {} - Validation Accuracy: {:.4f}".format(epoch+1, avg_val_accuracy))

        return train_loss, val_loss