# How to use `src/load_bert.py`?

Import the library:
```
from load_bert import load_all, load_config_and_tokenizer, load_model
```

Specify your model directory:
```
model_directory = /path/to/you/model
```

Specify the number of class labels and task type:
```
num_classes = 2
task_type = 'sequence_classification'
```

Load model, configuration, and tokenizer"
```
model, config, tokenizer = load_all(model_directory, num_classes, task_type)
```
    

