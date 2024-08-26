### Unit tests for the load_bert_test module ###

import os
import torch
import pytest

#from typing import override
from ..src.load_bert import BertCustomHead, load_all, load_config_and_tokenizer, load_model
from .fixtures import setUp, tearDown

@pytest.mark.usefixtures("setUp")
class TestBertLoad():

    # Use for specific test later
    outsideclass_tests_num_classes = 2
    outsideclass_tests_task_type = 'sequence_classification'

    def test_intializeBertWithNull(self, setUp):
        """
        Verify that BERT can be initialized with no config or tokenizer provided
        """

        config = None
        num_classes = 1
        task_type = None

        with pytest.raises(ValueError):
            bert = BertCustomHead(config, 1, task_type)
