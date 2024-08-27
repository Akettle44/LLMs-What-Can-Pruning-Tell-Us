### Unit tests for the load_bert_test module ###

import os
import pytest

from ..src.model import BertCustomHead
from ..src.load_bert import load_all, load_config_and_tokenizer, load_model

@pytest.mark.usefixtures("setUp")
class TestBertLoad():

    def test_LoadAll(self, setUp):
        """ 
        Verify that model, config, and tokenizer can all be loaded from disk
        NOTE: 
        This is dependent on your local environment. An instance of bert
        already needs to exist. If this test is failing that is likely why.
        TODO: Improve this to be inclusive of local testing module without tracking
        large files
        """

        config = None
        num_classes = 2
        task_type = 'sequence_classification'


        test_dir, root_dir, model_dir,  = setUp

        model_directory = os.path.join(model_dir, "bert_finetuned_sst2")

        bert, config, tokenizer = load_all(model_directory, num_classes, task_type)
        assert bert.config == config
        assert bert.num_classes == num_classes
        assert bert.task_type == task_type


