# LoRA-Prune: Hybrid Model Compression & Adaptation

**Author:** Muhammed E. Cham  
**Focus:** Large Language Model (LLM) Optimization & Efficiency

### 📌 Project Overview
LoRA-Prune is a hybrid framework designed to compress and adapt Large Language Models for resource-constrained environments. By combining **Structured Pruning** with **LoRA (Low-Rank Adaptation)**, this method reduces model size significantly while maintaining high performance.

### 🚀 Key Performance Metrics
* **Compression:** Pruned up to **25%** of the RoBERTa-base model.
* **Accuracy Preservation:** Achieved a negligible accuracy drop (~0.12%).
* **Efficiency:** Reduced total model size from ~125M parameters to ~94M (Light) and ~62.5M (Aggressive).

### 🛠 Technical Stack
* **Frameworks:** PyTorch, HuggingFace Transformers, PEFT, Accelerate.
* **Evaluation:** Scikit-learn, Evaluate.
* **Development:** AI-augmented workflow using GitHub Copilot for optimized refactoring.

### 📂 How to Execute
1. Install dependencies:
   `pip install transformers datasets torch accelerate evaluate peft scikit-learn`
2. Run an experiment (25% pruning):
   `python train.py --use_lora --pruning_amount 0.25`
