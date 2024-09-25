# Concrete implementation of taskDataset applied to SST2

from .dataset_interface import TaskDataset
from overrides import override

class Sst2Dataset(TaskDataset):
    def __init__(self, dataset_name, task_name, tokenizer, root_dir, loadLocal=False):
        super().__init__(dataset_name, task_name, tokenizer, root_dir, loadLocal)
        self.loadDataset()

    @override
    def preprocess(self, examples):
        return self.tokenizer(examples['sentence'], padding='max_length')

    @override
    def encode(self):
        self.tokenized_dataset_hf = self.dataset.map(self.preprocess, batched=True)
                                      
        self.tokenized_dataset_pt = self.tokenized_dataset_hf.with_format('torch', \
                                columns=['input_ids', 'attention_mask', 'label'])