# data.py handles ingesting and preparing the dataset for the different tasks

import os
import torch
import datasets

from abc import ABC, abstractmethod

class taskDataset(ABC):
    def __init__(self, dataset_name, task_name, tokenizer, max_length,
                 loadLocal=False, root_dir=os.path.dirname(os.getcwd())):

        self.dataset = None
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loadLocal = loadLocal
        self.root_dir = root_dir
        self.pretokenized = False

    def loadDataset(self):
        """ Load the dataset from hf or from local disk
        """
        if self.loadLocal:
            dataset_dir = os.path.join(os.path.join(self.root_dir, "datasets"), self.dataset_name)
            self.dataset = datasets.load_dataset(dataset_dir)
            self.pretokenized = True
        else:
            self.dataset = datasets.load_dataset(self.dataset_name, self.task_name)

    def saveDataset(self, write_path):
        """ Write an existing dataset in RAM to disk
        """
        if self.dataset is None:
            raise ValueError("taskDataset::saveDataset: Trying to save empty dataset")
        
        self.dataset.save_to_disk(write_path)

    @abstractmethod
    def preprocess(examples):
        """ Return tokenized outputs for particular examples.
            Typically not called directly. It is normally called
            via the map function from HuggingFace's dataset

        Args:
            examples: Particular rows from a dataset
        """
        pass

    @abstractmethod
    def encode():
        """ Perform the tokenization of the dataset for a 
            particular task
        """
        pass

    @abstractmethod
    def createDataLoaders(batch_size_train, batch_size_test):
        """ Create dataloaders for training (train, validation, test)
        """
        pass