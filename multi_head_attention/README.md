# Week 12 Day 2 — Multi-Head Attention & Positional Encoding

### Objective

Understand how Transformers capture multiple relationships simultaneously using Multi-Head Attention and how positional information is injected into a permutation-invariant attention mechanism.

---

### Concepts Learned

#### Multi-Head Attention

* One attention head can learn only limited relationships.
* Multiple heads learn different patterns simultaneously.
* Each head has independent:

  * Query weights
  * Key weights
  * Value weights
* Head dimension:

```python
d_k = d_model / num_heads
```

* Outputs from all heads are concatenated and projected using:

```python
MultiHead(Q,K,V)
=
Concat(head1,...,headh)W_O
```

---

#### Why Multiple Heads?

Different heads may focus on:

* Syntax
* Semantics
* Coreference
* Long-range dependencies
* Sentiment signals

simultaneously.

---

#### Permutation Invariance

Self-attention alone has no notion of token order.

Example:

```text
I love NLP
```

and

```text
NLP love I
```

contain identical tokens but different meaning.

Without positional information, self-attention cannot distinguish them.

---

#### Positional Encoding

Position information is added to token embeddings:

```python
Input =
TokenEmbedding +
PositionalEncoding
```

---

#### Sinusoidal Positional Encoding

Even dimensions:

```python
PE(pos,2i)
=
sin(pos/10000^(2i/d))
```

Odd dimensions:

```python
PE(pos,2i+1)
=
cos(pos/10000^(2i/d))
```

Advantages:

* No trainable parameters
* Works for arbitrary sequence lengths
* Encodes relative position information

---

#### Learned Positional Embeddings

* Position vectors are learned during training.
* Used by BERT.
* More flexible but limited by training context length.

---

#### RoPE (Rotary Position Embedding)

Instead of adding position vectors:

```text
Rotate Q and K vectors
```

Advantages:

* Better relative position modeling
* Better long-context extrapolation
* Widely used in modern LLMs

---

### Implemented

#### Head Splitting

Converted:

```python
(batch_size, seq_len, d_model)
```

to:

```python
(batch_size, num_heads, seq_len, d_k)
```

---

#### Positional Encoding

Implemented sinusoidal positional encoding using NumPy.

Generated:

```python
(seq_len, d_model)
```

position matrices.

---

### Key Takeaways

Today I learned:

* Why multiple attention heads are necessary
* How head splitting works
* Why d_k is reduced per head
* Why self-attention is permutation invariant
* How positional encodings solve the ordering problem
* Difference between sinusoidal, learned, and rotary positional embeddings
* Why RoPE is preferred in many modern LLMs

---
