# Concrete implementation of taskDataset applied to Cola

from .dataset_interface import TaskDataset
from overrides import override
from torch.utils.data import DataLoader

class ColaDataset(TaskDataset):
    def __init__(self, dataset_name, task_name, tokenizer, root_dir, loadLocal=False):
        super(TaskDataset, self, dataset_name, task_name, tokenizer, root_dir, loadLocal).__init__()
        self.loadDataset()
        self.max_length = self.findMaxLengthInDataset()

    @override
    def preprocess(self, examples):
        return self.tokenizer(examples['sentence'], padding='max_length', max_length=self.max_length)

    @override
    def encode(self):
        self.tokenized_dataset_hf = self.dataset.map(self.preprocess, batched=True, \
                                fn_kwargs={'tokenizer': self.tokenizer, 'max_len': self.max_length})
        self.tokenized_dataset_pt = self.tokenized_dataset_hf.with_format('torch', \
                                columns=['input_ids', 'token_type_ids', 'attention_mask'])

    def findMaxLengthInDataset(self):
        max_len_train = len(max(self.dataset['train']['sentence'][:]))
        max_len_val = len(max(self.dataset['validation']['sentence'][:]))
        max_len_test = len(max(self.dataset['test']['sentence'][:]))
        max_len = max(max_len_train, max_len_val, max_len_test)
        return max_len
        


