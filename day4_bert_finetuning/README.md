# Week 12 Day 4 — BERT Fine-Tuning Fundamentals

## Objective

Learn how pretrained Transformer models are used in real-world NLP applications and understand the complete fine-tuning pipeline using Hugging Face Transformers.

Unlike previous days that focused on Transformer internals, today's focus was on applying pretrained models such as BERT and DistilBERT for text classification tasks.

---

## Why This Topic Matters

Training a Transformer from scratch requires:

- Massive datasets
- Large GPU clusters
- Significant training time

Instead, modern NLP workflows use:

```text
Pretrained Model
↓
Task-Specific Dataset
↓
Fine-Tuning
↓
Deployment
```

Examples:

- Sentiment Analysis
- Spam Detection
- Intent Classification
- Document Classification
- Content Moderation
- Customer Feedback Analysis

---

## Concepts Learned

### 1. Hugging Face Transformers

The Hugging Face library provides easy access to pretrained Transformer models.

Main components:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

---

### 2. AutoTokenizer

Converts raw text into model inputs.

Responsibilities:

- Tokenization
- Vocabulary lookup
- Adding special tokens
- Padding
- Truncation
- Attention mask creation

Example:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

---

### 3. Tokenization Process

Input:

```text
I am learning AI
```

Tokens:

```text
['i', 'am', 'learning', 'ai']
```

Model Input:

```text
[CLS] i am learning ai [SEP]
```

---

### 4. input_ids

Neural networks cannot process text directly.

Tokenizer converts tokens into vocabulary indices.

Example:

```python
[101, 1045, 2572, 4083, 9932, 102]
```

Where:

```text
101 = [CLS]
102 = [SEP]
```

Purpose:

```text
Text → Token IDs → Embeddings
```

---

### 5. Attention Mask

Used to distinguish:

```text
Real Tokens
```

from:

```text
Padding Tokens
```

Example:

```python
[1,1,1,1,1,0,0]
```

Meaning:

```text
1 = Valid Token
0 = Padding
```

Purpose:

Prevent the attention mechanism from attending to padding tokens.

---

### 6. token_type_ids

Used when processing sentence pairs.

Example:

```text
Sentence A
Sentence B
```

Representation:

```python
[0,0,0,0,1,1,1,1]
```

Purpose:

Identify which token belongs to which sentence.

---

### 7. [CLS] Token

Special token placed at the beginning of every sequence.

Example:

```text
[CLS] The food was amazing
```

After multiple Transformer layers:

```text
[CLS]
```

contains information about the entire sentence.

Used for:

- Sentiment Analysis
- Text Classification
- Spam Detection
- Intent Classification

---

### 8. [SEP] Token

Special separator token.

Example:

```text
[CLS]
Sentence A
[SEP]
Sentence B
[SEP]
```

Purpose:

Separate multiple sentences in a single input sequence.

---

### 9. AutoModelForSequenceClassification

Loads:

```text
Pretrained Transformer
+
Classification Head
```

Example:

```python
from transformers import AutoModelForSequenceClassification

model =
AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)
```

Architecture:

```text
Input
↓
BERT
↓
CLS Representation
↓
Linear Layer
↓
Class Scores
```

---

### 10. Fine-Tuning Strategy

Pretrained models already contain language knowledge.

Large learning rates may destroy useful representations.

Recommended:

```python
2e-5
```

to

```python
5e-5
```

Purpose:

Gradually adapt the model to a new task.

---

### 11. AdamW Optimizer

Standard optimizer for BERT fine-tuning.

Example:

```python
optimizer =
torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)
```

Advantage:

Separates:

```text
Weight Decay
```

from:

```text
Gradient Updates
```

for better regularization.

---

### 12. DistilBERT

Compressed version of BERT.

Comparison:

| Model | Parameters |
|---------|---------|
| BERT Base | 110M |
| DistilBERT | 66M |

Benefits:

- Faster inference
- Lower memory usage
- Smaller model size
- Similar performance

DistilBERT achieves approximately:

```text
97% of BERT performance
```

while being significantly smaller.

---

### 13. Domain-Specific BERT Models

General BERT can be adapted to specific domains.

Examples:

#### FinBERT

Domain:

```text
Finance
```

Applications:

- Financial News
- Market Sentiment
- Earnings Reports

---

#### BioBERT

Domain:

```text
Healthcare
```

Applications:

- Medical Text
- Clinical Notes
- Biomedical Literature

---

#### LegalBERT

Domain:

```text
Law
```

Applications:

- Contracts
- Legal Documents
- Case Analysis

---

## Practical Implementation

Implemented:

### Tokenization

```python
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

---

### Sequence Classification Model

```python
model =
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)
```

---

### Forward Pass

```python
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=labels
)
```

---

### Loss Computation

```python
loss = outputs.loss
```

---

### Optimization

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

### Prediction

```python
prediction =
torch.argmax(
    outputs.logits,
    dim=1
)
```

---

## Key Takeaways

Today I learned:

- How Hugging Face simplifies Transformer usage
- How tokenization works in BERT
- Purpose of input_ids
- Purpose of attention_mask
- Purpose of token_type_ids
- How [CLS] represents an entire sentence
- How [SEP] separates sentences
- How sequence classification models work
- Why low learning rates are used during fine-tuning
- Why AdamW is the standard optimizer
- Why DistilBERT is commonly used in production
- How domain-specific BERT models are created

---

## Files

```text
day4_bert_finetuning/
│
├── tokenizer_demo.py
├── bert_classifier.py
├── distilbert_demo.py
└── README.md
```

---

## Next Step

### Week 12 Day 5

Topics:

- DistilBERT Fine-Tuning
- Review Classification Dataset
- Training Loop on Real Data
- Evaluation Metrics
- Model Comparison
- Preparation for Saturday Project