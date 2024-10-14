---
license: bsd-3-clause
datasets:
- AnishJoshi/nl2bash-custom
language:
- en
metrics:
- sacrebleu
base_model:
- Salesforce/codet5p-220m-bimodal
pipeline_tag: text2text-generation
tags:
- Python
- PyTorch
- Transformers
- english-to-bash
- nl2bash
- nl2cmd
---

# NL to Bash Translator

This model is a fine-tuned version of `codet5p-220m-bimodal` for translating natural language (NL) commands into Bash code. It simplifies command-line usage by allowing users to describe desired tasks in plain English and generates corresponding Bash commands.

## Model Overview

- **Task:** Natural Language to Bash Code Translation
- **Base Model:** codet5p-220m-bimodal
- **Training Focus:** Accurate command translation and efficient execution

## Dataset Description

The dataset used for training consists of natural language and Bash code pairs:

- **Total Samples:** 24,573
- **Training Set:** 19,658 samples
- **Validation Set:** 2,457 samples
- **Test Set:** 2,458 samples

Each sample contains:
- Natural language command (`nl_command`)
- Corresponding Bash code (`bash_code`)
- Serial number (`srno`)

## Training Setup

### Training Parameters

- **Learning Rate:** 5e-5
- **Batch Size:** 8 (training), 16 (evaluation)
- **Number of Epochs:** 5
- **Warmup Steps:** 500
- **Gradient Accumulation Steps:** 2
- **Weight Decay:** 0.01
- **Evaluation Strategy:** End of each epoch
- **Mixed Precision:** Enabled (FP16)

### Optimizer and Scheduler

- **Optimizer:** AdamW
- **Scheduler:** Linear learning rate with warmup

### Training Workflow

- Tokenization and processing to fit model input requirements
- Data Collator: `DataCollatorForSeq2Seq`
- Evaluation Metric: BLEU score

### Training Performance

| Epoch | Training Loss | Validation Loss | BLEU | Precision Scores | Brevity Penalty | Length Ratio | Translation Length | Reference Length |
|-------|---------------|-----------------|-------|----------------------------|-----------------|--------------|-------------------|------------------|
| 1 | 0.1882 | 0.1534 | 0.2751| [0.682, 0.516, 0.405, 0.335]| 0.5886 | 0.6536 | 26,316 | 40,264 |
| 2 | 0.1357 | 0.1198 | 0.3016| [0.731, 0.575, 0.470, 0.401]| 0.5684 | 0.6390 | 25,729 | 40,264 |
| 3 | 0.0932 | 0.1007 | 0.3399| [0.769, 0.629, 0.530, 0.464]| 0.5789 | 0.6465 | 26,032 | 40,264 |
| 4 | 0.0738 | 0.0889 | 0.3711| [0.795, 0.669, 0.582, 0.522]| 0.5851 | 0.6511 | 26,214 | 40,264 |
| 5 | 0.0641 | 0.0810 | 0.3939| [0.810, 0.700, 0.622, 0.566]| 0.5893 | 0.6541 | 26,336 | 40,264 |

### Test Performance

- **Test Loss:** 0.0867
- **Test BLEU Score:** 0.3699
- **Precision Scores:** [0.809, 0.692, 0.611, 0.555]
- **Brevity Penalty:** 0.5604
- **Length Ratio:** 0.6333
- **Translation Length:** 26,108
- **Reference Length:** 41,225

## Usage

### Load the Model and Tokenizer

from transformers import AutoTokenizer, AutoModel

# Option 1: Load from Hugging Face Hub
```python
model_name = "yuDivy/codet5p-220m-bimodal-finetune-english-to-bash" # Replace with the actual model name 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Option 2: Load from local directory
# local_model_path = "path/to/your/downloaded/model" # Replace with your local path
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModel.from_pretrained(local_model_path)
```
### Prepare Input

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set the model to evaluation mode

# Add the prefix to the input command
nl_command = "Your natural language command here"
input_text_with_prefix = f"bash: {nl_command}"

# Tokenize the input
inputs_with_prefix = tokenizer(input_text_with_prefix, return_tensors="pt", truncation=True, max_length=128).to(device)
```

### Generate Bash Code

```python
# Generate bash code
with torch.no_grad():
outputs_with_prefix = model.generate(
**inputs_with_prefix,
max_new_tokens=200,
num_return_sequences=1,
temperature=0.3,
top_p=0.95,
do_sample=True,
eos_token_id=tokenizer.eos_token_id,
)

generated_code_with_prefix = tokenizer.decode(outputs_with_prefix[0], skip_special_tokens=True)
print("Generated Bash Command:", generated_code_with_prefix)
```

## Example Outputs

Input: "bash: Enable the shell option 'cmdhist'"
Expected Output: `shopt -s cmdhist`
Generated Output: `shopt -s cmdhist`

## Language Bias and Generalization

The model exhibits some language bias, performing better when the natural language command closely matches training examples. Minor variations in output can occur based on command phrasing:

1. Original Command: "Find all files under /path/to/base/dir and change their permission to 644."
Generated Bash Code: `find /path/to/base/dir -type f -exec chmod 644 {} +`

2. Variant Command: "Modify the permissions to 644 for every file in the directory /path/to/base/dir."
Generated Bash Code: `find /path/to/base/dir -type f -exec chmod 644 {} \;`

The model generally captures the intended functionality, but minor variations in output can occur.

## Limitations and Future Work

1. **Bash Command Accuracy:** While the BLEU score and precision metrics are promising, some generated commands may still require manual refinement.
2. **Handling Complex Commands:** For highly complex tasks, the model may not always produce optimal results.
3. **Language Variation:** The model's performance might degrade if the input deviates significantly from the training data.