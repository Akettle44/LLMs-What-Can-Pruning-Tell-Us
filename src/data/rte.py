# Concrete implementation of taskDataset applied to RTE

from .dataset_interface import TaskDataset
from overrides import override

class RteDataset(TaskDataset):
    def __init__(self, dataset_name, task_name, tokenizer, root_dir, loadLocal=False):
        super(TaskDataset, self, dataset_name, task_name, tokenizer, root_dir, loadLocal).__init__()
        self.loadDataset()

    @override
    def preprocess(self, examples):
        return self.tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    @override
    def encode(self):
        self.tokenized_dataset_hf = self.dataset.map(self.preprocess, batched=True, \
                                      fn_kwargs={'tokenizer': self.tokenizer})
        self.tokenized_dataset_pt = self.tokenized_dataset_hf.with_format('torch', \
                                columns=['input_ids', 'attention_mask'])
                   
