# Week 12 Day 1 — Self-Attention From Scratch

## Objective

Understand the fundamental idea behind the Transformer architecture and implement Scaled Dot-Product Attention using NumPy.

---

## Why This Topic Matters

Self-attention is the core innovation introduced in the Transformer architecture in the paper:

Attention Is All You Need (2017)

Unlike RNNs and LSTMs, which process tokens sequentially, self-attention allows every token to interact with every other token simultaneously.

This enables:

- Parallel computation
- Better long-range dependency modeling
- Faster training on GPUs
- Foundation of BERT, GPT, T5, ViT, and modern LLMs

---

## Concepts Learned Today

### 1. Attention

Attention allows a token to determine which other tokens are most relevant when creating its representation.

Instead of remembering information through hidden states like LSTMs, a token can directly access information from any other token.

---

### 2. Query (Q), Key (K), and Value (V)

Each token is projected into three vectors:

#### Query

Represents:

"What information am I looking for?"

```python
Q = X @ WQ
```

#### Key

Represents:

"What information do I contain?"

```python
K = X @ WK
```

#### Value

Represents:

"What information should I return?"

```python
V = X @ WV
```

---

### 3. Dot Product Similarity

Queries are compared against Keys using a dot product.

```python
scores = Q @ K.T
```

A larger dot product indicates stronger similarity between two tokens.

---

### 4. Why Transpose K?

Without transpose:

```python
(3,4) @ (3,4)
```

Matrix multiplication is invalid.

With transpose:

```python
(3,4) @ (4,3)
```

Result:

```python
(3,3)
```

This allows every token to be compared against every other token.

---

### 5. Attention Score Matrix

For a sequence containing n tokens:

```python
scores.shape = (n,n)
```

Each element:

```text
scores[i,j]
```

represents how much token i attends to token j.

---

### 6. Softmax

Raw attention scores are converted into probabilities.

```python
weights = softmax(scores)
```

Properties:

- Values between 0 and 1
- Row sums equal 1
- Represents token importance

---

### 7. Numerical Stability in Softmax

Before applying exponential:

```python
x = x - np.max(x)
```

Purpose:

- Prevent overflow
- Keep probabilities unchanged
- Improve numerical stability

---

### 8. Why Use axis=-1?

```python
np.sum(exp, axis=-1)
```

Softmax should operate across each token's attention scores.

Each row should become a valid probability distribution.

Without axis=-1, probabilities from different tokens would mix together.

---

### 9. Why Divide by √d_k?

Scaled Attention:

```python
scores = scores / np.sqrt(d_k)
```

Reason:

As vector dimensions increase, dot products become larger.

Large values cause softmax saturation.

Scaling keeps values stable and gradients useful during training.

---

### 10. Softmax Saturation

Example:

```python
softmax([100,2,1])
```

Produces:

```text
[0.999..., 0.000..., 0.000...]
```

The model becomes overly confident and gradients become very small.

Scaling by √d_k prevents this problem.

---

### 11. Contextual Embeddings

The same word can have different meanings depending on context.

Example:

```text
bank loan
```

vs

```text
river bank
```

Self-attention allows the model to generate different embeddings for the same word in different contexts.

---

### 12. Self-Attention vs Cross-Attention

#### Self-Attention

Q, K, and V come from the same sequence.

```python
Q = XWQ
K = XWK
V = XWV
```

#### Cross-Attention

Used in encoder-decoder architectures.

```python
Q = Decoder
K = Encoder
V = Encoder
```

The decoder attends to encoder outputs.

---

## Attention Formula Derived Today

Step 1:

```python
scores = Q @ K.T
```

Step 2:

```python
scores = scores / np.sqrt(d_k)
```

Step 3:

```python
weights = softmax(scores)
```

Step 4:

```python
output = weights @ V
```

Final Equation:

```python
Attention(Q,K,V)
=
softmax(QK.T / √d_k)V
```

---

## Implementation

Implemented:

- Softmax from scratch
- Scaled Dot-Product Attention
- Shape verification
- Numerical stability handling

---

## Sample Shapes

```python
Q.shape = (5,64)
K.shape = (5,64)
V.shape = (5,64)
```

Output:

```python
scores.shape = (5,5)
weights.shape = (5,5)
output.shape = (5,64)
```

---

## Key Takeaways

Today I learned:

- How self-attention works internally
- Why Q, K, and V are needed
- How attention scores are computed
- Why K is transposed
- Why softmax is applied
- Why scaling by √d_k is necessary
- How contextual embeddings are created
- Difference between self-attention and cross-attention
- How the Transformer computes token relationships

---

## Next Step

Week 12 Day 2

- Multi-Head Attention
- Positional Encoding
- Why Transformers need positional information