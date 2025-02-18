import torch
import os
import matplotlib.pyplot as plt
from data.factory import TaskFactory
from data.cola import ColaDataset
from data.sst2 import Sst2Dataset
from model.model import BertCustom
from training.train import PtTrainer
from pruning.pruners import L1Pruner
from model.load_save import saveModelToDisk, loadModelFromDisk
from utils.utils import Utils

# Trainer for models
def train():

    # Paths
    root_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(root_dir, 'models')

    # Task (determines bert and dataset config)
    task_name = "cola"

    # Load Hyperparameters
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(model_dir, 'hyps'), task_name + '.txt'))

    # Model
    config = None
    num_classes = 2
    tokenizer = None
    task_type = 'sequence_classification'
    bert = BertCustom(config, num_classes, tokenizer, task_type)
    # Freezes encoder layers
    bert.freezeLayers(hyps['freezeLayers'])
    model_name = 'cola-training-04'

    # Dataset
    dataset_name = "glue"
    loadLocal = False
    dataset = TaskFactory.createTaskDataSet(dataset_name, task_name, bert.tokenizer, root_dir, loadLocal)

    # Encode Dataset
    batch_size_train = hyps['trbatch']
    batch_size_val = hyps['valbatch']
    dataset.encode()
    dataset.createDataLoaders(batch_size_train, batch_size_val, hyps['numworkers'], True, True, subset_percentages=[1, 1, 1])

    trainer = PtTrainer(bert, dataset)
    trainer.sendToDevice()
    trainer.setHyps(hyps)
    trainer.updateOptimizerLr()
    tr_loss, tr_acc, val_loss, val_acc = trainer.fineTune()
    saveModelToDisk(trainer.model, root_dir, model_name)

    # Plot loss results (show it decreases)
    save_path = os.path.join(root_dir, "models/sequence_classification/" + model_name)
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

# Prune + Evaluate using CKA
def eval():
    # Paths
    root_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(root_dir, "models")
    specific_model = os.path.join(model_dir, "sequence_classification/cola-training-04")

    # Load BERT
    bert, hyps = loadModelFromDisk(specific_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset_name = "glue"
    task_name = "cola"
    root_dir = os.getcwd()
    loadLocal = False
    dataset = TaskFactory.createTaskDataSet(dataset_name, task_name, bert.tokenizer, root_dir, loadLocal)

    # Encode Dataset
    batch_size_train = hyps['trbatch']
    batch_size_val = hyps['valbatch']
    dataset.encode()
    dataset.createDataLoaders(batch_size_train, batch_size_val, hyps["numworkers"], True, True, subset_percentages=[1, 1, 1])

    # Define save directory
    base_save_dir = os.path.join(os.path.join(os.path.dirname(os.getcwd()), "plots"))
    dirs = sorted(os.listdir(base_save_dir))
    i = int(dirs[-1].split('_')[-1])
    i += 1
    run = "run_" + str(i)
    save_dir = os.path.join(base_save_dir, run)
    os.mkdir(save_dir)

    # Define pruner + prune models
    pr = L1Pruner(bert, save_dir)
    pr.prune()
    accs = pr.compareModels(dataset.train_loader, device)
    print(accs)

if __name__ == "__main__":
    #train()
    eval()