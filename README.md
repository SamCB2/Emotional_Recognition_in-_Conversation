# Emotion Recognition in Conversations using a Teacher-Student Framework

## Overview

In the domain of Natural Language Processing (NLP), understanding human dialogue is a critical research challenge. Tasks like **emotion recognition**, **intent classification**, and **sentiment analysis** are essential for building conversational systems that approach human-level understanding.

This project focuses on **utterance-level emotion classification** using the **DailyDialog** dataset. It explores and compares three models:

- **RoBERTa Large (BERT-ERC)** — as a teacher baseline  
- **Bidirectional LSTM (bcLSTM)** — as a student baseline  
- A **Teacher-Student framework** — where RoBERTa Large guides bcLSTM using knowledge distillation

The ultimate goal is to build an efficient yet accurate emotion recognition system that balances contextual richness with computational efficiency.

---

## Dataset: DailyDialog

- **Source**: [DailyDialog Dataset](http://yanran.li/dailydialog)  
- **Structure**: Multi-turn dialogues on everyday topics like travel, health, and work  
- **Emotion Classes**: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise  
- **Utterances**:  
  - Training: ~80,000  
  - Validation: ~10,000 
  - Test: ~10,000  

---

## Models

### 1. RoBERTa Large (BERT-ERC) — Teacher Baseline

A fine-tuned RoBERTa-large model for Emotion Recognition in Conversations (ERC). Key steps:

- **Input Formatting**:
  ```
  <s> Speaker A <mask> says: [Utterance] </s>
  ```
  Utilizes speaker tokens and a mask token to encode structure and role.

- **Segmentation**:  
  Token embeddings are segmented into **past**, **query**, and **future**, mean-pooled and concatenated for fine-grained context representation.

- **Prediction**:  
  A fully connected layer with dropout generates logits for the 7 emotion classes.

### 2. Bidirectional LSTM (bcLSTM) — Student Baseline

A lightweight, standalone model that captures bidirectional context in dialogue:

- **Architecture**:
  - Embedding layer  
  - Bidirectional LSTM  
  - Mean pooling over hidden states  
  - Fully connected layer with dropout  

- **Advantage**:  
  No pretraining required, much fewer parameters than transformer models

### 3. Teacher-Student Framework — Proposed Model

Combines the contextual power of RoBERTa with the lightweight efficiency of bcLSTM:

- **Teacher**: RoBERTa-large fine-tuned to generate contextual logits ("soft labels") and hidden state embeddings  
- **Student**: bcLSTM trained using a mix of:  
  - **Supervised loss** (cross-entropy on ground-truth labels)  
  - **Distillation loss** (KL divergence from teacher's logits)  

This setup allows the student to **inherit** contextual knowledge while remaining compact and efficient.

---

## Implementation

### File Structure

- `Part1_BERT_ERC_Teacher_Model.ipynb` — Implements RoBERTa-large baseline  
- `Part2_bcLSTM_Student_Model.ipynb` — Implements standalone bcLSTM model  
- `Part3_Teacher_Student_Proposed_Model.ipynb` — Implements teacher-student architecture  

### Training Details

- **Optimizer**: AdamW  
- **Dropout**: 0.3 in student  
- **Hardware**: Google Colab (T4 GPU), ~12 hours for full training  
- **Checkpoints**: Saved after every epoch (model state, optimizer, loss, logits)  

---

## Results

| Model                 | Accuracy (Test) | Training Accuracy | F1-Score (Weighted, excl. Neutral) |
|----------------------|------------------|--------------------|------------------------------------|
| RoBERTa Large (Teacher) | 83.95%          | —                  | ~0.6142 (per [1])                  |
| bcLSTM (Student)        | 84.42%          | —                  | ~0.4116 (per [2])                  |
| Teacher-Student (Ours)  | **85.47%**      | 87.59%             | **0.5774**                         |

### Observations

- **Neutral** class dominates and is easiest to classify (F1 ~91.66%)  
- **Happiness** performs moderately (F1 ~59.03%)  
- **Fear**, **Disgust**, and **Anger** are underrepresented in the dataset and perform poorly  
- **Teacher-Student model** performs better than both baselines and closely approaches state-of-the-art  

---

## How to Run

Ensure you have a compatible GPU (A100 preferred, T4 acceptable), and run the notebooks in order:

1. `Part1_BERT_ERC_Teacher_Model.ipynb`  
2. `Part2_bcLSTM_Student_Model.ipynb`  
3. `Part3_Teacher_Student_Proposed_Model.ipynb`  

**Dependencies**:
- `transformers`  
- `torch`  
- `sklearn`  
- `numpy`  
- `pandas`

---

## Conclusions

- The **teacher-student framework** delivers superior performance and computational efficiency  
- **bcLSTM** shows promise when guided by richer contextual embeddings  
- The model **approaches state-of-the-art** performance despite a highly imbalanced dataset  
- Future work can address class imbalance and explore advanced attention mechanisms  

---

## References

1. Qin et al., *“Emotion Recognition in Conversations with Transformer and Structured Commonsense Knowledge”*, ACL 2021  
2. Ghosal et al., *“DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation”*, ACL 2019  


---

## Contact

**Authors**:  
- Samer Meleka   
**Email**: [samermmeleka@gmail.com](mailto:samermmeleka@gmail.com)
