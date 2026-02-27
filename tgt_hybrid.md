# Section III: Methodology – TGT Hybrid Model

## 1. Problem Formulation

Given a user $u$ and their interaction history $S_u = [i_1, i_2, ..., i_L]$, the next-item prediction task aims to recommend item $i_{L+1}$ that the user will interact with next. The objective is to maximize the predicted score $s_{ui}$ for candidate item $i$:

$$
\text{argmax}_{i} \, s_{ui}
$$

where $s_{ui}$ is the predicted relevance score.

## 2. BERT4Rec Backbone

- **Embedding:** Each item $i$ is mapped to an embedding $e_i$; positions are mapped to $p_j$.
- **Input:** $X = [e_{i_1} + p_1, ..., e_{i_L} + p_L]$
- **Self-Attention Blocks:**
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
  Multi-head self-attention and feed-forward layers encode the sequence.
- **Scoring:** The final sequence representation $h_u$ is used to score candidate items via dot product.

## 3. Temporal Graph Transformer (TGT) Encoder

- **Graph Construction:** Build item graph $G = (V, E)$ from user histories, incorporating temporal information (timestamps).
- **Time Encoding:** Each interaction timestamp is projected to a continuous time embedding.
- **Temporal Attention:**
  $$
  \text{Attention}(Q, K, V, T) = \text{softmax}\left(\frac{QK^T + \text{TimeBias}(T)}{\sqrt{d_k}}\right)V
  $$
  where $T$ is the time embedding and $\text{TimeBias}(T)$ is a learned bias from time encoding.
- **GCN Update Rule:**
  $$
  E^{(\ell)} = \sigma\left(\hat{A} E^{(\ell-1)} W^{(\ell)}\right)
  $$
  where $E^{(\ell)}$ is the item embedding at layer $\ell$, $\hat{A}$ is the normalized adjacency matrix, $W^{(\ell)}$ is the layer weight, and $\sigma$ is an activation function.

## 4. Gated Fusion (Novelty)

- **Fusion Function:**
  $$
  \alpha = \text{sigmoid}(\text{gate})
  $$
  where $\text{gate}$ is a learnable parameter.
- **Fused Embedding:**
  $$
  h_i(u) = \alpha \, h_i^{\text{BERT}} + (1 - \alpha) \, h_i^{\text{TGT}}
  $$
  where $h_i^{\text{BERT}}$ is the BERT4Rec embedding, $h_i^{\text{TGT}}$ is the TGT embedding, and $\alpha$ is a learnable fusion weight. This allows the model to adaptively combine sequential and temporal graph signals.

## 5. Training Objective

- **Loss:** Binary Cross-Entropy (BCE) loss with negative sampling.
  $$
  \mathcal{L} = -\sum_{(u, i^+)} \log \sigma(s_{ui^+}) - \sum_{(u, i^-)} \log (1 - \sigma(s_{ui^-}))
  $$
  where $i^+$ is a positive item, $i^-$ is a sampled negative item, and $\sigma$ is the sigmoid function.

## 6. Model Architecture Figure

Below is a schematic diagram of the TGT Hybrid Model pipeline:

```
sequence (user history) + timestamps
      |
      v
+-------------------+
|   TGT Encoder     |
+-------------------+
      |
      v
+-------------------+
|   BERT4Rec        |
+-------------------+
      |
      v
+-------------------+
|   Gated Fusion    |
+-------------------+
      |
      v
+-------------------+
|     Scoring       |
+-------------------+
      |
      v
  Recommendation
```

---

_This file provides a detailed methodology for the TGT Hybrid Model, suitable for Section III of an IEEE paper._
