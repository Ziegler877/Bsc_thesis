# Analysis of Foundational Models for Semantic Authorship Attribution
[cite_start]**Author:** Daniel Ziegler [cite: 1]  
[cite_start]**Matriculation Number:** 11909868 [cite: 2]  
[cite_start]**Thesis:** Bachelor's Thesis Proposal [cite: 4]

## 📌 Project Description
[cite_start]This research evaluates the effectiveness of Foundational Models (FMs) for the task of closed-set semantic authorship attribution[cite: 37]. [cite_start]The goal is to determine if models can capture author-specific stylistic and semantic signatures rather than relying on traditional stylometric n-grams[cite: 10, 15].

## 🧪 Experimental Setup
[cite_start]The project compares different model families using a few-shot attribution methodology[cite: 40, 48].

### Models Analyzed:
* [cite_start]**Encoder Family:** `multilingual-e5-small`, `multilingual-e5-large`[cite: 43].
* [cite_start]**LLM Family:** `Llama-2 7B`, `Llama-Maveric`[cite: 46, 47].
* [cite_start]**Techniques:** Metric learning and Low-Rank Adaptation (LoRA) evaluation[cite: 50, 56].

### Datasets:
* [cite_start]**High-Volume Challenge:** `Reuter_50_50`[cite: 53].
* [cite_start]**Low-Resource Challenge:** `MiniDarkReddit`[cite: 55].

## 📊 Evaluation Metrics
Performance is assessed using:
* [cite_start]Accuracy.
* [cite_start]F1 Score (Micro and Macro).
* [cite_start]Log-Loss.

## 🛠 Setup & Installation
1. Clone this repository.
2. Ensure you have a virtual environment active: `python -m venv .venv`.
3. Install dependencies: `pip install -r requirements.txt`.