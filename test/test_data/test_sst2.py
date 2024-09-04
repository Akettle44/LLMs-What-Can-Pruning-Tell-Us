### Unit test SST2 Dataset 

import os
import pytest
import shutil

from ...src.data.sst2 import Sst2Dataset
from ...src.model.model import BertCustom

@pytest.mark.usefixtures("setUp")
class TestSst2Dataset():

    def testInit(self, setUp):
        """ Verify that dataset initializes properly
        """

        dataset_name = "glue"
        task_name = "cola"
        tokenizer = None
        root_dir = os.getcwd()
        loadLocal = False
        sst2 = Sst2Dataset(dataset_name, task_name, tokenizer, root_dir, loadLocal)

        assert sst2.dataset_name == dataset_name
        assert sst2.task_name == task_name
        assert sst2.tokenizer == tokenizer
        assert sst2.root_dir == root_dir
        assert sst2.loadLocal == loadLocal
        assert sst2.dataset is not None

    def testEncoding(self, setUp):
        """ Verify that the dataset is encoded properly
        """

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

        # Encoding
        sst2.encode()
        sst2.createDataLoaders(8, 1, 1)

        # Check Dataloaders
        item = next(iter(sst2.train_loader))
        assert set(item.keys()) == set(['input_ids', 'attention_mask', 'label'])



