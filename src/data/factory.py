### This file provides the object instantiation for different types

from .cola import ColaDataset
from .sst2 import Sst2Dataset

class TaskFactory():

    @staticmethod
    def createTaskDataSet(dataset_name, task_name, tokenizer, root_dir, loadLocal):
        match task_name:
            case "cola":
                return ColaDataset(dataset_name, task_name, tokenizer, root_dir, loadLocal)
            case "sst2":
                return Sst2Dataset(dataset_name, task_name, tokenizer, root_dir, loadLocal)
            case _:
                raise ValueError(f"Task: {task_name} is not currently supported")