import torch
import os
from data.factory import TaskFactory
from data.cola import ColaDataset
from data.sst2 import Sst2Dataset
from model.model import BertCustom
from training.train import PtTrainer

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
    sst2.createDataLoaders(batch_size_train, batch_size_val, 1)

    trainer = PtTrainer(bert, sst2)
    trainer.sendToDevice()
    trainer.setHyps(epochs=1)
    out = trainer.fineTune()


if __name__ == "__main__":
    main()