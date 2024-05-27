# File for training the various BERT models using huggingface and different tasks.
# Right now this is specific to the cola task using BERT for sequence classification

import os
import torch
import torch.nn.utils.prune as prune
import datasets
import transformers
import numpy as np
import matplotlib.pyplot as plt

# TODO: Find non-global solutions for these
METRIC = None
TOKENIZER = None

# TODO: Make this an object?
def chooseTaskHF(task):
    global METRIC
    dataset = datasets.load_dataset("glue", task)
    METRIC = datasets.load_metric("glue", task)
    return dataset

def preprocessTokens(examples):
    return TOKENIZER(examples['sentence'], truncation=True)

def tokenizeDataset(model_checkpoint, dataset):
    global TOKENIZER
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    encoded_dataset = dataset.map(preprocessTokens, batched=True)
    return encoded_dataset

def loadModel(model_checkpoint, task, **kwargs):
    # TODO: This will eventually become task specific
    num_labels = kwargs['num_labels']
    model = transformers.BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return METRIC.compute(predictions=predictions, references=labels)
 
def createTrainer(model, checkpoint, task, dataset, metric_name, save_dir, **hyps):
                  
    model_name = checkpoint.split("/")[-1]
    output_name = f"{model_name}-finetuned-{task}"
    output_dir = os.path.join(save_dir, output_name)

    # Unpack hyperparameters
    epochs = hyps['epochs']
    batch_size = hyps['batch_size']
    lr = hyps['learning_rate']
    wd = hyps['weight_decay']

    args = transformers.TrainingArguments(
        output_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=wd,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
    )

    trainer = transformers.Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=TOKENIZER,
        compute_metrics=compute_metrics
    )

    return trainer

def fineTuneModel(trainer):
    trainer.train()

def evaluateFineTuning(trainer):
    results = trainer.evaluate()
    return results

def main():
    
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "models")
    # LOCAL
    #specific_model = os.path.join(model_dir, "your_model_here")
    # From HF
    checkpoint = "bert-base-uncased"

    # Constants
    task = "cola"
    num_labels = 2
    batch_size = 16
    learning_rate = 2e-5
    epochs = 5
    weight_decay = 0.01
    metric_name = "matthews_correlation"

    # Grab dataset
    dataset = chooseTaskHF(task)
    encoded_dataset = tokenizeDataset(checkpoint, dataset)

    # Grab model
    model = loadModel(checkpoint, task, num_labels=num_labels)

    trainer = createTrainer(model, checkpoint, task, encoded_dataset, metric_name, model_dir, \
                  epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, \
                  weight_decay=weight_decay)

    fineTuneModel(trainer)
    results = evaluateFineTuning(trainer)

    trainer.save_model(os.path.join(model_dir, f"bert-uncased-{task}-local-train"))

if __name__ == '__main__':
    main()

