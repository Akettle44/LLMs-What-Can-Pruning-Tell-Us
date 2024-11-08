# File to explore pruning on BERT models. TODO: Generalize this file 

import os
import torch
import torch.nn.utils.prune as prune
import datasets
import transformers
import numpy as np
import matplotlib.pyplot as plt
import hf_train as local_trainer

# Calculate sparisty of the network
def calcSparsity(model):
    zeros = 0
    elements = 0

    for _, param in model.named_parameters():

        zero_count = torch.sum(param==0.0).item()
        zeros += zero_count

        element_count = param.numel()
        elements += element_count

    if(elements == 0):
        return 0

    return zeros / elements

# Note: The pruning strength is for the subset of the neurons that you want to prune (defined by the paramater list created in the build parameter list function #
def buildParameterList(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        # Linear: Includes linear layers in attention head as well
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    return parameters_to_prune

# NOTE: Metric is defined globally
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) # TODO: switch to torch?
    return local_trainer.METRIC.compute(predictions=predictions, references=labels)

def buildTrainerEval(model, encoded_dataset, tokenizer, save_dir, **hyps):

    batch_size = hyps["batch_size"]
    metric_name = hyps["metric_name"]
    output_name = os.path.join(save_dir, "Eval-Directory-Not-Used")

    args = transformers.TrainingArguments(
        output_name,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model=metric_name,
        push_to_hub=False,
    )

    evaluator = transformers.Trainer(
        model,
        args,
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    return evaluator

def evaluateUnstructuredPruningOnBert(model_directory, dataset, tokenizer, save_dir, **options):

    # Unpack options
    pruning_strength = options['pruning_percentages']
    num_labels = options['num_labels']
    metric_name = options['metric_name']
    batch_size = options['batch_size']

    pruning_results = []
    # Try different pruning strenghts
    for p_strength in pruning_strength:

        # Load model
        test_model = transformers.BertForSequenceClassification.from_pretrained(model_directory, num_labels=num_labels)

        # Choose layers to prune
        pruning_params = buildParameterList(test_model)

        if(p_strength > 0):

            # Perform pruning
            prune.global_unstructured(
                pruning_params,
                pruning_method=prune.L1Unstructured,
                amount=p_strength
            )

            # Make pruning permenant
            for module,name in pruning_params:
                prune.remove(module, name)

        sparsity = calcSparsity(test_model)
        print(f"Pruning strength: {p_strength}, Sparsity: {sparsity}")

        # Create evaluator
        evaluator = buildTrainerEval(test_model, dataset, tokenizer, save_dir, metric_name=metric_name, \
                                     batch_size=batch_size)

        # Evaluate performance on BERT 
        results = evaluator.evaluate()
        metric_res = results['eval_matthews_correlation']

        pruning_results.append((sparsity, metric_res))
        
    return pruning_results

def plotPruningResults(pruning_results):

    # Plot results
    sparsities = []
    metric_results = []

    for sparsity, met in pruning_results:
        sparsities.append(sparsity)
        metric_results.append(met)

    plt.plot(sparsities, metric_results, label="Matthews Coefficient", color="blue")
    plt.title("Matthew's Coefficient as a Function of Network Sparsity")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Matthews Coefficient (Higher Better)")
    plt.legend()
    plt.show();

def main():

    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "models")

    checkpoint = os.path.join(model_dir, "bert-uncased-cola-local-train")
    num_labels = 2
    task = "cola"
    metric_name = "matthews_correlation"

    # Grab dataset
    dataset = local_trainer.chooseTaskHF(task)
    encoded_dataset = local_trainer.tokenizeDataset(checkpoint, dataset)
    
    # TODO: eventually add arguments to specify this
    pruning_percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    batch_size = 16

    pruning_results = evaluateUnstructuredPruningOnBert(checkpoint, encoded_dataset, local_trainer.TOKENIZER, \
                                                        save_dir=model_dir, num_labels=num_labels, pruning_percentages=pruning_percentages, \
                                                        batch_size=batch_size, metric_name=metric_name)
    
    plotPruningResults(pruning_results)

if __name__ == '__main__':
    main()

