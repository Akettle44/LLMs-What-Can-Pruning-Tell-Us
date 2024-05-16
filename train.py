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


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train_loader, val_loader = load_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        
        optimizer.zero_grad()
        
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    print("Epoch {} - Average training loss: {:.4f}".format(epoch+1, avg_train_loss))
    
    model.eval()
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch
            
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            val_accuracy += (logits.argmax(axis=1) == label_ids).mean().item()
    
    avg_val_accuracy = val_accuracy / len(val_loader)
    print("Epoch {} - Validation Accuracy: {:.4f}".format(epoch+1, avg_val_accuracy))
