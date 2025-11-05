# LoRA-Prune: A Hybrid Approach for Compressing and Adapting Language Models

A project by **Muhammed E Cham (22314170)** that demonstrates how to make Large Language Models smaller and more efficient.

## Abstract

This paper proposes **LoRA-Prune**, a hybrid method that first compresses a base model using pruning and then adapts it using LoRA. Our experiments show that we can prune up to 25% of the `roberta-base` model with a negligible drop in accuracy (~0.12%), significantly reducing the total model size for deployment.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install transformers datasets torch accelerate evaluate peft scikit-learn
    ```
2.  **Run an experiment (e.g., 25% pruning):**
    ```bash
    python train.py --use_lora --pruning_amount 0.25
    ```

## Results on SST-2

| Experiment Name         | Pruning Amount | Total Model Size (Approx.) | Accuracy |
| ----------------------- | :------------: | :------------------------: | :------: |
| **Baseline: LoRA**      |       0%       |         ~125 Million       | 91.86%   |
| **LoRA-Prune (Light)**  |      25%       |          ~94 Million       | 91.74%   |
| **LoRA-Prune (Aggressive)**|      50%       |         ~62.5 Million      | 91.40%   |
