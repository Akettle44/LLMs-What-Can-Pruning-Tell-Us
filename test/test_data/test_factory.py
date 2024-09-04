### This file provides unit tests for the task factory in src/data

import pytest
import os
from ...src.data.factory import TaskFactory
from ...src.data.cola import ColaDataset
from ...src.data.sst2 import Sst2Dataset

@pytest.mark.usefixtures("setUp")
class TestTaskFactory():

    def testCola(self, setUp):
        """ Test that the COLA object is instantiated properly
        """
        dataset_name = "glue"
        task_name = "cola"
        tokenizer = None
        root_dir = os.getcwd()
        loadLocal = False
        taskDs = TaskFactory.createTaskDataSet(dataset_name, task_name, tokenizer, root_dir, loadLocal)
        assert isinstance(taskDs, ColaDataset)

    def testSst2(self, setUp):
        """ Test that the Sst2 object is instantiated properly
        """

        dataset_name = "glue"
        task_name = "sst2"
        tokenizer = None
        root_dir = os.getcwd()
        loadLocal = False
        taskDs = TaskFactory.createTaskDataSet(dataset_name, task_name, tokenizer, root_dir, loadLocal)
        assert isinstance(taskDs, Sst2Dataset)