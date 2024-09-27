import torch
import os
import matplotlib.pyplot as plt
from data.factory import TaskFactory
from data.cola import ColaDataset
from data.sst2 import Sst2Dataset
from model.model import BertCustom
from training.train import PtTrainer
from model.load_save import saveModelToDisk, loadModelFromDisk

# Trainer for models
def main():

    # Model
    config = None
    num_classes = 2
    tokenizer = None
    task_type = 'sequence_classification'
    bert = BertCustom(config, num_classes, tokenizer, task_type)

    # Dataset
    dataset_name = "glue"
    task_name = "sst2"
    root_dir = os.getcwd()
    loadLocal = False
    sst2 = Sst2Dataset(dataset_name, task_name, bert.tokenizer, root_dir, loadLocal)

    # Encode Dataset
    batch_size_train = 8
    batch_size_val = 8
    sst2.encode()
    sst2.createDataLoaders(batch_size_train, batch_size_val, 8, True, True, subset_percentages=[1, 1, 1])

    trainer = PtTrainer(bert, sst2)
    trainer.sendToDevice()
    trainer.setHyps(epochs=3)
    tr_loss, tr_acc, val_loss, val_acc = trainer.fineTune()
    saveModelToDisk(trainer.model, os.path.dirname(root_dir), "sst-training-5-wd-0.01")

    # Plot loss results (show it decreases)
    plt.plot(range(len(tr_loss)), tr_loss, label="Training Loss", color='blue')
    plt.plot(range(len(val_loss)), val_loss, label="Validation Loss", color='orange')
    plt.title(f"Training curves for {task_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show();

if __name__ == "__main__":
    main()