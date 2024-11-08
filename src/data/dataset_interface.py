# data.py handles ingesting and preparing the dataset for the different tasks

import os
import datasets
import numpy as np
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod

class TaskDataset(ABC):

    @abstractmethod
    def __init__(self, dataset_name, task_name, tokenizer, root_dir, loadLocal=False):
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.root_dir = root_dir
        self.loadLocal = loadLocal
        self.dataset = None
        self.tokenized_dataset_hf = None
        self.tokenized_dataset_pt = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.pretokenized = False

    def getTokenizationStatus(self):
        return self.pretokenized

    def loadDataset(self):
        """ Load the dataset from hf or from local disk
        """
        if self.loadLocal:
            dataset_dir = os.path.join(os.path.join(self.root_dir, "datasets"), self.dataset_name)
            self.dataset = datasets.load_dataset(dataset_dir)
            self.pretokenized = True
        else:
            self.dataset = datasets.load_dataset(self.dataset_name, self.task_name)

    def saveRawDataset(self, write_path):
        """ Write an existing dataset in RAM to disk
        """
        if self.dataset is None:
            raise ValueError("taskDataset::saveRawDataset: Trying to save empty dataset")
        
        self.dataset.save_to_disk(write_path)

    def saveTokenizedDataset(self, write_path):
        """ Write an existing dataset in RAM to disk
        """
        if self.tokenized_dataset_hf is None:
            raise ValueError("taskDataset::saveEncodedDataset: Trying to save empty dataset")
        self.tokenized_dataset_hf.save_to_disk(write_path)

    @abstractmethod
    def preprocess(self, examples):
        """ Return tokenized outputs for particular examples.
            Typically not called directly. It is normally called
            via the map function from HuggingFace's dataset

        Args:
            examples: Particular rows from a dataset
        """
        pass

    @abstractmethod
    def encode(self):
        """ Perform the tokenization of the dataset for a 
            particular task
        """
        pass

    def createDataLoaders(self, batch_size_train, batch_size_test, num_workers, pin_memory=False, shuffle=True, subset_percentages=[1, 1, 1]):
        """ Create dataloaders for training (train, validation, test)
            Subset percentages: Multiplier for choosing the size of the dataset subset
        """

        train_ds = self.grabRandomSubset(self.tokenized_dataset_pt['train'], subset_percentages[0])
        valid_ds = self.grabRandomSubset(self.tokenized_dataset_pt['validation'], subset_percentages[1])
        test_ds = self.grabRandomSubset(self.tokenized_dataset_pt['test'], subset_percentages[2])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size_train, num_workers=num_workers, \
                                       shuffle=shuffle, pin_memory=pin_memory)
        self.val_loader = DataLoader(valid_ds, batch_size=batch_size_train, num_workers=num_workers, \
                                    shuffle=shuffle, pin_memory=pin_memory)
        # For the experiments in this repo, the test batch size typically needs to be 1
        self.test_loader = DataLoader(test_ds, batch_size=batch_size_test, num_workers=num_workers, \
                                     shuffle=shuffle, pin_memory=pin_memory)

    def grabRandomSubset(self, dataset, percentage):
        """ Take a subset of the dataset using random indices in that range
        """
        numrange = len(dataset)
        k = int(numrange * percentage)
        idxs = np.random.choice(numrange, k, replace=False)
        return dataset.select(idxs)




