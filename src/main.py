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
    # Freezes encoder layers
    bert.freezeLayers([6, 7, 8, 9, 10])
    model_name = "sst-training-17-dp-0.4-fz6-10"

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
    trainer.setHyps(epochs=5)
    tr_loss, tr_acc, val_loss, val_acc = trainer.fineTune()
    print(tr_acc, val_acc)
    saveModelToDisk(trainer.model, os.path.dirname(root_dir), model_name)

    # Plot loss results (show it decreases)
    save_path = os.path.join(os.path.dirname(root_dir), "models/sequence_classification/" + model_name)
    plt.plot(range(len(tr_loss)), tr_loss, label="Training Loss", color='blue')
    plt.plot(range(len(val_loss)), val_loss, label="Validation Loss", color='orange')
    plt.title(f"Training curves for {task_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))

    # Plot loss results (show it decreases)
    plt.figure()
    plt.plot(range(len(tr_acc)), tr_acc, label="Training Accuracy", color='blue')
    plt.plot(range(len(val_acc)), val_acc, label="Validation Accuracy", color='orange')
    plt.title(f"Training curves for {task_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_path, "accuracy_plot.png"))


if __name__ == "__main__":
    main()