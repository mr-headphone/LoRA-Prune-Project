
import argparse
import json
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import evaluate
from peft import get_peft_model, LoraConfig
import torch.nn.utils.prune as prune

def parse_args():
    parser = argparse.ArgumentParser(description="Run LoRA and LoRA-Prune experiments.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the base model.")
    parser.add_argument("--task_name", type=str, default="sst2", help="GLUE task name.")
    parser.add_argument("--output_dir", type=str, default="/content/LoRA-Prune_Project/results", help="Directory to save results.")
    parser.add_argument("--pruning_amount", type=float, default=0.0, help="Fraction of weights to prune (0.0 means no pruning).")
    parser.add_argument("--use_lora", action="store_true", help="Flag to apply LoRA.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    return parser.parse_args()

def prune_model(model, amount):
    print(f"--- Starting Unstructured L1 Pruning with amount: {amount} ---")
    parameters_to_prune = []
    for layer in model.roberta.encoder.layer:
        parameters_to_prune.append((layer.attention.self.query, 'weight'))
        parameters_to_prune.append((layer.attention.self.key, 'weight'))
        parameters_to_prune.append((layer.attention.self.value, 'weight'))
        parameters_to_prune.append((layer.attention.output.dense, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    print("--- Pruning complete and made permanent. ---")
    return model

def main():
    args = parse_args()
    print(f"Loading dataset '{args.task_name}' and tokenizer for '{args.model_name}'")
    raw_datasets = load_dataset("glue", args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(f"Loading base model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if args.pruning_amount > 0:
        model = prune_model(model, args.pruning_amount)
    if args.use_lora:
        print("--- Applying LoRA ---")
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.05, bias="none", task_type="SEQ_CLS")
        model = get_peft_model(model, lora_config)
        print("LoRA parameters:")
        model.print_trainable_parameters()
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    experiment_name = f"{args.model_name}_{args.task_name}"
    experiment_name += f"_prune{int(args.pruning_amount*100)}" if args.pruning_amount > 0 else ""
    experiment_name += "_lora" if args.use_lora else ""
    output_dir = os.path.join(args.output_dir, experiment_name)
    training_args = TrainingArguments(output_dir=output_dir, learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=args.epochs, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, report_to="none",)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets["train"], eval_dataset=tokenized_datasets["validation"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
    print(f"\n--- Starting Training for Experiment: {experiment_name} ---")
    trainer.train()
    print("\n--- Evaluating on the test set ---")
    eval_results = trainer.evaluate()
    print(eval_results)
    results_path = os.path.join(output_dir, "final_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"\nTraining complete. Final results saved to {results_path}")

if __name__ == "__main__":
    main()
