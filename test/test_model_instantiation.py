### Unit tests for the load_bert_test module ###
import pytest
from ..src.model import BertCustomHead
from .fixtures import setUp

@pytest.mark.usefixtures("setUp")
class TestBertInstantiation():

    # Use for specific test later
    def test_intializeBertWithNull(self):
        """
        Verify that BERT can be initialized with no config or tokenizer provided
        """

        config = None
        num_classes = 1
        task_type = None

        with pytest.raises(ValueError):
            bert = BertCustomHead(config, num_classes, task_type)

    def test_intializeBertSequence(self):
        """
        Verify that BERT can be initialized with sequence mode
        """

        config = None
        num_classes = 2
        task_type = 'sequence_classification'
        bert = BertCustomHead(config, num_classes, task_type)
        
        assert bert.config == config
        assert bert.num_classes == num_classes
        assert bert.task_type == task_type

    def test_intializeBertToken(self):
        """
        Verify that BERT can be initialized with Token mode
        """

        config = None
        num_classes = 2
        task_type = 'token_classification'
        bert = BertCustomHead(config, num_classes, task_type)
        
        assert bert.config == config
        assert bert.num_classes == num_classes
        assert bert.task_type == task_type

    def test_intializeBertMcq(self):
        """
        Verify that BERT can be initialized with Mcq mode
        """

        config = None
        num_classes = 2
        task_type = 'multiple_choice'
        bert = BertCustomHead(config, num_classes, task_type)
        
        assert bert.config == config
        assert bert.num_classes == num_classes
        assert bert.task_type == task_type

