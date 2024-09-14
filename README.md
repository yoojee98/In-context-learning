# In-context Learning Project

## 1 Introduction

### 1.1 Large Language Model and In-context Learning?
Large language models, such as GPT-3 and its successors, represent a significant leap forward in the field of artificial intelligence, specifically within the realm of natural language processing. These models are trained on vast datasets encompassing a wide swath of human knowledge, allowing them to generate text that can mimic human writing styles with remarkable accuracy. A key feature of these models is their ability to perform "in-context learning", which means they can understand and respond to new instructions or information presented within the immediate context of their input text. This capability enables a form of dynamic adaptation to new tasks without the need for explicit retraining or fine-tuning on specific datasets. As a result, large language models have found applications across a diverse range of fields, from automated content creation to sophisticated dialogue systems, showcasing their versatility and the power of advanced neural network architectures.


### 1.2 Multi-choice Question Answering
The task of multiple-choice question answering is a vital tool in assessing the reasoning capabilities of large language models. It involves presenting the model with a question alongside several possible answers, only one of which is correct. This approach is particularly beneficial as it gauges a model's ability to comprehend, deduce, and apply knowledge acquired during its training to new situations. Notably, initiatives like the Massive Multitask Language Understanding (MMLU) and the AI2 Reasoning Challenge (ARC) are instrumental in this context. They are designed to challenge a wide array of knowledge and reasoning skills, spanning science and mathematics to literature and common sense reasoning. For this assignment, we exclusively utilize the ARC dataset, which allows for rigorous, automatic evaluation of a model's reasoning capabilities.


## 2 Setup


### 2.1 Working Locally

If you have GPU resources on your personal computer, preparing for local work requires installing GPU drivers, CUDA, cuDNN, and PyTorch. Although completing the assignment without a GPU is feasible, model inference will be significantly slower. Therefore, having a GPU offers a substantial advantage for enhancing performance throughout the assignment.

### 2.2 Creating Python Environments

**Installing Python**: The code for this assignment has been verified with Python version 3.10.9.

**Virtual Environment**: The use of a virtual environment via Anaconda is recommended for this project. If you decide against using a virtual environment, it is your responsibility to ensure all dependencies are installed. To establish a Conda virtual environment, execute the following commands:

    conda create -n nlp_env python=3.10.9
    conda activate nlp_env

Follow the official PyTorch installation guidelines to set up the PyTorch environment. This guide uses PyTorch version 2.0.1 with CUDA 11.7. Alternate versions may be specified by adjusting the version number:

    pip install torch==2.0.1

Proceed to install sentence_transformers, transformers, accelerate, and str2bool:

    pip install sentence_transformers==2.3.1
    pip install transformers==4.31.0
    pip install accelerate==0.22.0
    pip install str2bool


## 3 Working on the Assignment


### 3.1 Basis of In-Context Learning

In-context learning enables large language models to adapt to and perform new tasks based on the examples and instructions embedded directly within the input prompt. This feature exploits the model's extensive pre-training on diverse text to infer the task requirements from the given context and generate a suitable output without additional training.

To illustrate, let's consider teaching a model to answer simple arithmetic questions through in-context learning. Below is a complete prompt that includes contextual cues (examples of arithmetic operations and their answers) and a target question for the model to answer:

```
Input Prompt:
- Example 1: What is 3 plus 2? Answer: 3 plus 2 is 5.
- Example 2: What is 6 minus 1? Answer: 6 minus 1 is 5.
- Question: What is 5 plus 7?

```

In this prompt, the first two lines are the contextual cues, demonstrating how the model should interpret and respond to arithmetic questions. The "Example 1" and "Example 2" parts show the model the format of the question and the expected format of the answer. The line starting with "Question" is the target question we want the model to answer, based on the pattern it learned from the contextual cues. By analyzing the input prompt, the model learns to map a new arithmetic input ("What is 5 plus 7?") to the appropriate output, leveraging the format and knowledge presented in the examples.


To prevent data leakage and maintain the validity of evaluations in in-context learning, it's important that the examples provided as contextual cues are not taken from the test set. These examples should instead be selected from the training dataset or be manually created. Selecting examples carefully is essential for effectively applying the model's extensive pre-trained knowledge.

### 3.2 Task Description
The multiple-choice question answering task involves presenting a model with a question and a set of possible answers, among which only one is correct.

To illustrate, consider the following example:

    Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
    Candidate Answers: (A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion
    Gold Answer: A

In this example, the model is tasked with analyzing the question and the provided candidate answers to determine which option correctly answers the question.

Throughout this assignment, you'll employ in-context learning to boost question-answering accuracy using the ARC dataset. Key to success is the adept formatting of examples and the strategic selection from the training data, ensuring the model effectively leverages its pre-trained knowledge. 


### 3.3 Get Code

The provided codebase is structured to facilitate conducting experiments with the ARC dataset. Here's an overview of the directory and file organization:

```
└── src
    ├── data
    │   ├── ARC-Challenge-test.jsonl                 # Test set for the ARC Challenge
    │   ├── ARC-Challenge-train.jsonl                # Training set for the ARC Challenge
    │   ├── ARC-Challenge-validation.jsonl           # Validation set for the ARC Challenge
    │   ├── ARC-Easy-test.jsonl                      # Test set for the ARC Easy
    │   ├── ARC-Easy-train.jsonl                     # Training set for the ARC Easy
    │   └── ARC-Easy-validation.jsonl                # Validation set for the ARC Easy
    ├── cache_utils.py                               # Utility functions for caching
    ├── modeling_attn_mask_utils.py                  # Utilities for attention mask modeling
    ├── download.py                                  # Script to download necessary models
    ├── configuration_phi.py                         # Defines the large language model (LLM) configuration
    ├── modeling_phi.py                              # Implements the LLM architecture
    ├── tokenization_codegen.py                      # Tokenizer definition for the model
    ├── eval_fewshot.py                              # Main script for conducting experiments
    ├── eval_fewshot_multigpu.py                     # Supports experiments using multiple GPUs
    ├── acc.py                                       # Calculates accuracy metric
    └── README.md                                    

```

To run experiments and compute the accuracy metric, the following commands can be used:

```
python eval_fewshot.py --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" --model path_of_llm --embedder path_of_embedder --start_index 0 --end_index 9999 --max_len 1024 --output_path path_to_save_model_predictions --overwrite False --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False
python acc.py --prediction_path path_to_save_model_predictions
```

Parameters explained:
- `model` and `embedder`: Specify paths to the large language model (e.g., Phi-1.5) and the embedding model (e.g., bge-small-en-v1.5), respectively. These models can be prepared with the `download.py` script.
- `data_path`: Path to the JSONL file containing the dataset for evaluation.
- `device_id`: GPU device IDs for running the experiments.
- `max_len`: Maximum sequence length for the model input.
- `output_path`: Directory to save model predictions.
- `prompt_type`: Version of the prompt formatting.
- `N`: Number of examples to include in the prompt.
- `top_k`: Whether to select the top k similar examples based on embedding similarity for in-context learning.
- `top_k_reverse`: Determines the order of appending similar examples, where the most similar example is placed closest to the question at hand.

Use `eval_fewshot_multigpu.py` in place of `eval_fewshot.py` for splitting the workload across multiple GPUs if running the model on a single GPU is not feasible.


## Reference

1. Textbooks Are All You Need II: phi-1.5 technical report
2. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
3. UNIFIEDQA: Crossing Format Boundaries with a Single QA System
4. Extending Context Window of Large Language Models via Positional Interpolation

