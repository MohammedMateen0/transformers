# Week 12 Day 3 — Transformer Encoder Block From Scratch

## Objective

Build and understand the complete Transformer Encoder Block that forms the foundation of modern NLP models such as BERT, GPT, RoBERTa, DeBERTa, LLaMA, Mistral, and many other Large Language Models.

The goal of this day was to understand how Multi-Head Attention, Residual Connections, Layer Normalization, and Feed Forward Networks work together inside a Transformer layer.

---

## Why This Topic Matters

Self-Attention alone is not enough to build a Transformer.

A complete Transformer layer contains:

- Multi-Head Attention
- Residual Connections
- Layer Normalization
- Feed Forward Network (FFN)

Understanding this architecture is essential for understanding:

- BERT
- GPT
- T5
- LLaMA
- Mistral
- Modern LLMs

---

## Transformer Encoder Architecture

```text
Input
 ↓
Multi-Head Attention
 ↓
Add & LayerNorm
 ↓
Feed Forward Network
 ↓
Add & LayerNorm
 ↓
Output
```

This block is stacked multiple times to build large Transformer models.

Examples:

- BERT Base → 12 Encoder Blocks
- BERT Large → 24 Encoder Blocks

---

## Concepts Learned

### 1. Feed Forward Network (FFN)

Formula:

```python
FFN(x)
=
max(0, xW₁ + b₁)W₂ + b₂
```

Architecture:

```text
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Output
```

Example:

```text
512
 ↓
2048
 ↓
512
```

### Purpose

Attention mixes information between tokens.

FFN performs non-linear transformations inside each token representation.

Key idea:

```text
Attention = communication between tokens

FFN = thinking inside each token
```

---

### 2. Residual Connections

Formula:

```python
Output = F(x) + x
```

Purpose:

- Improve gradient flow
- Reduce vanishing gradients
- Enable deeper Transformer networks

Residual connections act as a shortcut path that allows information and gradients to pass directly through the network.

---

### 3. Layer Normalization

LayerNorm normalizes features for each token independently.

Example tensor:

```python
(batch_size, sequence_length, d_model)

(32,100,512)
```

Normalization happens across:

```python
d_model
```

dimension.

Implementation:

```python
mean = x.mean(dim=-1, keepdim=True)

var = x.var(dim=-1, keepdim=True)

x_norm =
(x - mean) /
sqrt(var + eps)
```

### Purpose

- Stabilizes training
- Prevents activation values from becoming too large
- Improves optimization

---

### 4. BatchNorm vs LayerNorm

#### BatchNorm

Normalizes across:

```text
Batch Dimension
```

#### LayerNorm

Normalizes across:

```text
Feature Dimension
```

LayerNorm is preferred for NLP because sequence lengths and batch sizes can vary.

---

### 5. Pre-LN vs Post-LN

#### Post-LN (Original Transformer)

```text
Attention
 ↓
Add
 ↓
LayerNorm
```

#### Pre-LN (Modern LLMs)

```text
LayerNorm
 ↓
Attention
 ↓
Add
```

Examples:

- LLaMA → Pre-LN
- Mistral → Pre-LN

### Why Pre-LN?

Provides more stable gradients and easier optimization for very deep Transformer networks.

---

### 6. Encoder

Encoder uses:

```text
Bidirectional Attention
```

Each token can attend to:

- Previous tokens
- Future tokens

Example:

```text
The bank near the river...
```

The word:

```text
bank
```

can use both left and right context to understand meaning.

Models:

- BERT
- RoBERTa
- DeBERTa

Best suited for:

- Classification
- Sentiment Analysis
- NER
- Search

---

### 7. Decoder

Decoder uses:

```text
Causal Masking
```

Each token can only attend to previous tokens.

Example:

```text
I love machine _____
```

The model can see:

```text
I
love
machine
```

but not the next word.

Models:

- GPT
- LLaMA
- Mistral

Best suited for:

- Text Generation
- Chatbots
- Code Generation

---

### 8. BERT's [CLS] Token

BERT adds:

```text
[CLS]
```

at the beginning of every sequence.

Example:

```text
[CLS] The food was amazing
```

After passing through Transformer layers, the [CLS] token becomes a representation of the entire sentence.

Used for:

- Sentiment Classification
- Spam Detection
- Text Classification

---

## Implementation

Implemented from scratch using PyTorch:

### FeedForward Network

```python
class FeedForward(nn.Module)
```

Components:

- Linear Layer
- ReLU Activation
- Linear Layer

---

### TransformerBlock

```python
class TransformerBlock(nn.Module)
```

Implemented:

- Query Projection
- Key Projection
- Value Projection
- Scaled Dot-Product Attention
- Output Projection (W_O)
- Residual Connections
- Layer Normalization
- Feed Forward Network

---

## Attention Formula Used

```python
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

Where:

- Q = Query Matrix
- K = Key Matrix
- V = Value Matrix
- dₖ = Key Dimension

---

## Shape Verification

Input:

```python
(32,100,512)
```

Q, K, V:

```python
(32,100,512)
```

Attention Output:

```python
(32,100,512)
```

FFN Output:

```python
(32,100,512)
```

Final Output:

```python
(32,100,512)
```

The Transformer block preserves input dimensionality.

---

## Key Learnings

Today I learned:

- Complete Transformer Encoder architecture
- Purpose of Feed Forward Networks
- Why residual connections are necessary
- Difference between BatchNorm and LayerNorm
- Pre-LN vs Post-LN architectures
- Encoder vs Decoder architectures
- Why GPT uses causal masking
- Why BERT uses bidirectional attention
- Purpose of the [CLS] token
- How attention, normalization, and FFNs work together inside a Transformer block

---

## Files

```text
day3_transformer_encoder/
│
├── feed_forward.py
├── transformer_block.py
├── test_transformer.py
└── README.md
```

---

## Next Step

Week 12 Day 4

Topics:

- BERT vs GPT in depth
- Causal Masking implementation
- Decoder-only architecture
- Encoder-only architecture
- Why one masking decision creates GPT instead of BERT