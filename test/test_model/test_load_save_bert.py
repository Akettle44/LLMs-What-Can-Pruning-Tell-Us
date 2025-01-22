### Unit tests for the load_bert_test module ###

import os
import pytest
import shutil

from ...src.model.model import BertCustom
from ...src.model.load_save import saveModelToDisk, loadModelFromDisk

@pytest.mark.usefixtures("setUp")
class TestBertIO():

    @pytest.mark.order(1)
    def test_saveModel(self, setUp):
        """ Create and Save BERT model
        """

        config = None
        num_classes = 2
        tokenizer = None
        task_type = 'sequence_classification'
        test_dir, root_dir, model_dir,  = setUp

        # Create the model
        bert = BertCustom(config, num_classes, tokenizer, task_type, False)
        # Save the model to disk
        saveModelToDisk(bert, root_dir, "unit_test-bert")

    @pytest.mark.order(2)
    def test_loadModel(self, setUp):
        """ Load model 
        """

        # Create same model from saveModel

        test_dir, root_dir, model_dir, = setUp
        config = None
        num_classes = 2
        tokenizer = None
        task_type = 'sequence_classification'
        test_dir, root_dir, model_dir,  = setUp

        saved_bert = BertCustom(config, num_classes, tokenizer, task_type, False)

        # Load BERT from disk
        load_path = os.path.join(os.path.join(model_dir, task_type), "unit_test-bert")
        bert, _ = loadModelFromDisk(load_path)

        assert saved_bert.num_classes == bert.num_classes
        assert saved_bert.task_type == bert.task_type

        # Verify that model was correctly loaded
        assert saved_bert.state_dict().keys() == bert.state_dict().keys()

    @pytest.fixture(scope='class', autouse=True)
    def cleanUpTest(self, setUp):

        _, _, model_root_dir, = setUp
        yield # Allow test to run first
        
        model_name = "unit_test-bert"
        task_dir = os.path.join(model_root_dir, "sequence_classification")
        model_dir = os.path.join(task_dir, model_name)

        # Remove dummy model directory
        shutil.rmtree(model_dir)