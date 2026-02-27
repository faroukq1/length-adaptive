# Section III: Methodology – Hybrid Fixed Model

## 1. Problem Formulation

Given a user $u$ and their interaction history $S_u = [i_1, i_2, ..., i_L]$, the goal is next-item prediction: recommend item $i_{L+1}$ that the user will interact with next. Formally, for each user $u$, maximize the score $s_{ui}$ for candidate item $i$:

$$
\text{argmax}_{i} \, s_{ui}
$$

where $s_{ui}$ is the predicted relevance score.

## 2. SASRec Backbone

- **Embedding:** Each item $i$ is mapped to an embedding $e_i$; positions are mapped to $p_j$.
- **Input:** $X = [e_{i_1} + p_1, ..., e_{i_L} + p_L]$
- **Self-Attention Blocks:**
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
  Each block applies multi-head self-attention and position-wise feed-forward layers.
- **Scoring:** The final sequence representation $h_u$ is used to score candidate items via dot product.

## 3. Item Co-Occurrence Graph & GNN Encoder

- **Graph Construction:** Build item graph $G = (V, E)$ from user histories. Edges connect items co-occurring within a sliding window $w$.
- **Edge Weights:** Edge $(i, j)$ weight is proportional to co-occurrence frequency.
- **GCN Update Rule:**
  $$
  E^{(\ell)} = \sigma\left(\hat{A} E^{(\ell-1)} W^{(\ell)}\right)
  $$
  where $E^{(\ell)}$ is the item embedding at layer $\ell$, $\hat{A}$ is the normalized adjacency matrix, $W^{(\ell)}$ is the layer weight, and $\sigma$ is an activation function.

## 4. Length-Adaptive Fusion (Novelty)

- **Bins:** Define $L^*_{short}$ and $L^*_{long}$ to categorize user history lengths.
- **Fusion Function:**
  $$
  \alpha(u) = \begin{cases}
    \alpha_{short} & \text{if } L \leq L^*_{short} \\
    \alpha_{mid} & \text{if } L^*_{short} < L \leq L^*_{long} \\
    \alpha_{long} & \text{if } L > L^*_{long}
  \end{cases}
  $$
- **Fused Embedding:**
  $$
  h_i(u) = \alpha(u) e_i + (1 - \alpha(u)) g'_i
  $$
  where $e_i$ is the SASRec embedding, $g'_i$ is the GNN embedding, and $\alpha(u)$ adapts based on user history length. This allows the model to dynamically balance sequential and graph signals.

## 5. Training Objective

- **Loss:** Binary Cross-Entropy (BCE) loss with negative sampling.
  $$
  \mathcal{L} = -\sum_{(u, i^+)} \log \sigma(s_{ui^+}) - \sum_{(u, i^-)} \log (1 - \sigma(s_{ui^-}))
  $$
  where $i^+$ is a positive item, $i^-$ is a sampled negative item, and $\sigma$ is the sigmoid function.

## 6. Model Architecture Figure

Below is a schematic diagram of the Hybrid Fixed Model pipeline:

```
sequence (user history)
      |
      v
+-------------------+
|  Fused Embeddings |
+-------------------+
      |
      v
+-------------------+
|   Transformer     |
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

_This file provides a detailed methodology for the Hybrid Fixed Model, suitable for Section III of an IEEE paper._
