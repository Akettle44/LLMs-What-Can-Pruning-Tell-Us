import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification
from data import load_data

"""
This file (train.py) does the following -
(1) Imports the `bert-base-uncased` model from the transformers library for text classification
(2) Initialises the loss function and optimizer
(3) Imports the training and validation data from `data.py`
(4) Trains the model and returns the training losses and validation accuracies for each epoch

"""

def setupTraining():

    # Create model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # Send model to GPU (Have to do this before creating optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Set hyperparameters
    hyps = {}
    hyps['epochs'] = 3

    return model, device, loss_func, optimizer, hyps

def setupData():
    train_loader, val_loader = load_data()
    return train_loader, val_loader

def fineTune():
    
    # Setup various parameters of training
    model, device, loss_func, optimizer, hyps = setupTraining()
    train_loader, val_loader = setupData()

    # Unpack hyperparameters
    num_epochs = hyps['epochs']

    train_loss = []
    val_loss = []

    # Perform training
    for epoch in range(num_epochs):

        ### TRAINING ###
        model.train() # Set training mode in PyTorch
        epoch_train_loss = []
        
        # One iteration over dataset
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch

            optimizer.zero_grad()
            
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

def main():
    train_loss, val_loss = fineTune()
    print(train_loss)

if __name__ == '__main__':
    main()
