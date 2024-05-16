import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_data():
    """
    This function initialises a simple dataset for text classification that the BERT model can be trained on.
    Class labels: 0 and 1 (0 stands for negative and 1 stands for positive)
    Outputs: 
        (1) train_loader (This variable is the DataLoader object for the training set)
        (2) val_loader (This variable is the DataLoader object for the validation set)
    """

    texts = ["This is such a good product. I absolutely love it!!.", "This item sucks, its really bad.", 
    "I've been using this for a long time and I'm really happy about it",
    "The product has helped me a lot and its very inexpensive",
    "It's a ripoff and it broke in just a couple days, don't buy this item guys!"]
    labels = [1, 0, 1, 1, 0] 

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, 
                            max_length = 64,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.1)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_loader = DataLoader(val_data, batch_size=32)

    return train_loader, val_loader
