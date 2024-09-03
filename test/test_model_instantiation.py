### Unit tests for the load_bert_test module ###

import pytest
import transformers
from ..src.model import BertCustom
from .conftest import setUp

@pytest.mark.usefixtures("setUp")
class TestBertInstantiation():

    # Use for specific test later
    def test_intializeBertWithNull(self):
        """
        Verify that BERT can be initialized with no config or tokenizer provided
        """

        config = None
        num_classes = 1
        tokenizer = None
        task_type = None

        with pytest.raises(ValueError):
            bert = BertCustom(config, num_classes, tokenizer, task_type)

    def test_intializeBertSequence(self):
        """
        Verify that BERT can be initialized with sequence mode
        """

        config = None
        num_classes = 2
        task_type = 'sequence_classification'
        tokenizer = None
        bert = BertCustom(config, num_classes, tokenizer, task_type)
        
        assert not bert.config == None
        assert bert.num_classes == num_classes
        assert not bert.tokenizer == None
        assert bert.task_type == task_type

    def test_intializeBertToken(self):
        """
        Verify that BERT can be initialized with Token mode
        """

        config = None
        num_classes = 2
        tokenizer = None
        task_type = 'token_classification'
        bert = BertCustom(config, num_classes, tokenizer, task_type)
        
        assert not bert.config == None
        assert bert.num_classes == num_classes
        assert not bert.tokenizer == None
        assert bert.task_type == task_type

    def test_intializeBertMcq(self):
        """
        Verify that BERT can be initialized with Mcq mode
        """

        config = None
        num_classes = 2
        tokenizer = None
        task_type = 'multiple_choice'
        bert = BertCustom(config, num_classes, tokenizer, task_type)
        
        assert not bert.config == None
        assert bert.num_classes == num_classes
        assert not bert.tokenizer == None
        assert bert.task_type == task_type

    def test_correctParamsPretrained(self):
        """ Verify that bert-base-uncased is loaded by default
        """

        config = None
        num_classes = 2
        tokenizer = None
        task_type = 'sequence_classification'
        model_name = 'bert-base-uncased'

        bert = transformers.BertModel.from_pretrained(model_name, num_classes)
        bertcus = BertCustom(config, num_classes, tokenizer, task_type)

        # Verify that parameters are the same
        assert str(bertcus.model.state_dict()) == str(bert.state_dict())

    def test_pretrainedNotUsed(self):

        """ Verify that non-pretrained version doesn't use pretrained weights
        """

        config = None
        num_classes = 2
        tokenizer = None
        task_type = 'sequence_classification'
        model_name = 'bert-base-uncased'

        bert = transformers.BertModel.from_pretrained(model_name, num_classes)
        bertcus = BertCustom(config, num_classes, tokenizer, task_type, False)

        # Verify that parameters are the same
        assert not str(bertcus.model.state_dict()) == str(bert.state_dict())

