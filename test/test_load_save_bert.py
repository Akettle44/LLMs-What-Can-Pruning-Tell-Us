### Unit tests for the load_bert_test module ###

import os
import pytest

from ..src.model import BertCustom
from ..src.load_save import save_model_to_disk, load_model_from_disk

@pytest.mark.usefixtures("setUp")
class TestBertIO():

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
        save_model_to_disk(bert, root_dir, "unit-test-bert")

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
        load_path = os.path.join(os.path.join(model_dir, task_type), "unit-test-bert")
        bert = load_model_from_disk(load_path)

        assert saved_bert.confg == bert.config
        assert saved_bert.num_classes == bert.num_classes
        assert saved_bert.tokenizer == bert.tokenizer
        assert saved_bert.task_type == bert.task_type

        # Verify that parameters are the same
        assert str(saved_bert.model.state_dict()) == str(bert.state_dict())