### Unit test Cola Dataset 

import os
import pytest
import shutil

from ...src.data.cola import ColaDataset
from ...src.model.model import BertCustom

@pytest.mark.usefixtures("setUp")
class TestColaDataset():

    def testInit(self, setUp):
        """ Verify that dataset initializes properly
        """

        dataset_name = "glue"
        task_name = "cola"
        tokenizer = None
        root_dir = os.getcwd()
        loadLocal = False
        cola = ColaDataset(dataset_name, task_name, tokenizer, root_dir, loadLocal)

        assert cola.dataset_name == dataset_name
        assert cola.task_name == task_name
        assert cola.tokenizer == tokenizer
        assert cola.root_dir == root_dir
        assert cola.loadLocal == loadLocal
        assert cola.dataset is not None
        assert cola.max_length != 0 

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
        task_name = "cola"
        root_dir = os.getcwd()
        loadLocal = False
        cola = ColaDataset(dataset_name, task_name, bert.tokenizer, root_dir, loadLocal)

        # Encoding
        cola.encode()
        cola.createDataLoaders(8, 1, 1)

        # Check Dataloaders
        item = next(iter(cola.train_loader))
        assert set(item.keys()) == set(['input_ids', 'token_type_ids', 'attention_mask', 'label'])



