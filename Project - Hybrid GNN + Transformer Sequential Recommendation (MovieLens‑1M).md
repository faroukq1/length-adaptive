# Length-Adaptive Hybrid GNN + SASRec for MovieLens-1M

Goal: Build a **length-adaptive hybrid GNN + SASRec** model for next-item prediction on MovieLens-1M, and deliver clean results + notes to your teacher before 1 March.

---

## Phase 0 – Setup & Notes

- [x] Create Obsidian vault page for the project (`SR-Hybrid-GNN-Transformer-ML1M`). ✅ 2026-02-05
- [x] Create sub-notes: `01_problem_scope`, `02_datasets`, `03_baselines`, `04_model_math`, `05_experiments`, `06_results_for_teacher`. ✅ 2026-02-05
- [x] Setup code repo (Git): `sr-hybrid-gnn-transformer-ml1m`. ✅ 2026-02-05
- [x] Add basic folders: `data/`, `src/`, `src/models/`, `src/data/`, `src/train/`, `src/experiments/`, `configs/`, `notebooks/`. ✅ 2026-02-05
- [x] Create `requirements.txt` or `pyproject.toml` (PyTorch, PyG or DGL if needed, RecBole/other libs optional). ✅ 2026-02-05

---

## Phase 1 – Problem scope & research angle

Note: All of this goes into `01_problem_scope` in Obsidian.

- [ ] Write precise **task definition**:
    - [ ] "Next-item sequential recommendation on MovieLens-1M using implicit feedback."
    - [ ] Explain: input = ordered movie sequence per user, output = probability distribution over next movie.
- [ ] Define **research angle**:
    - [ ] Hybrid GNN (global item graph) + Transformer (SASRec) for sequential recommendation.
    - [ ] Novelty: **length-adaptive fusion** between GNN and SASRec item embeddings based on user history length.
- [ ] Draft a short **research gap** (3–5 sentences):
    - [ ] State that standard SASRec models ignore global item graphs.
    - [ ] State that existing GNN+Transformer hybrids typically use **fixed fusion** for all users.
    - [ ] Propose your length-adaptive fusion (short-history users rely more on GNN, long-history more on sequence).
    - [ ] Mention evaluation on MovieLens-1M next-item prediction with standard metrics.
    - [ ] Note any teacher constraints (if any) in this note.

---

## Phase 2 – Dataset & preprocessing (MovieLens-1M)

Note: Document in `02_datasets`.

- [ ] Download **MovieLens-1M** (1M ratings).
    - [ ] Record URL and license.
    - [ ] Save raw data in `data/ml-1m/raw/`.
- [ ] Implement parsing script `src/data/ml1m_preprocess.py`:
    - [ ] Read `ratings.dat` (or equivalent file).
    - [ ] Map original user ids and movie ids to internal integer indices (`uid`, `iid`).
    - [ ] Convert ratings to implicit feedback (interaction if rating ≥ threshold, e.g. 3 or 4).
- [ ] Build **chronological sequences**:
    - [ ] Group by user, sort by timestamp.
    - [ ] Filter users with fewer than N interactions (e.g. N=5).
    - [ ] Save sequences as list of item ids per user.
- [ ] Define **train/val/test**:
    - [ ] Use leave-one-out or similar: last item for test, second-last for validation, rest for training.
    - [ ] Implement logic to generate (user, sequence_prefix, target_item) pairs for training/val/test.
- [ ] Implement **negative sampling** (for training/eval):
    - [ ] For each positive target, sample K negative items not in user's history.
- [ ] Save processed datasets to `data/ml-1m/processed/` (e.g., `.pkl` or `.pt`).
- [ ] In `02_datasets`, document: data filters, split strategy, negative sampling, and final counts (users, items, interactions).

---

## Phase 3 – Baseline models (SASRec, optional GRU4Rec)

Note: Document in `03_baselines`.

- [x] Decide framework path: ✅ 2026-02-05
    - [x] Either reuse an existing SASRec implementation (Kang's repo / ezSASRec / RecBole). ✅ 2026-02-05
    - [x] Or write a minimal SASRec from scratch (if you want more control). ✅ 2026-02-05
- [x] Integrate MovieLens-1M data loader: ✅ 2026-02-05
    - [x] Implement `src/data/dataloader_ml1m.py` that outputs batches of sequences and targets for SASRec. ✅ 2026-02-05
    - [x] Ensure correct masking, padding, and max sequence length. ✅ 2026-02-05
- [x] Implement or adapt **SASRec**: ✅ 2026-02-05
    - [x] Item embeddings, positional embeddings. ✅ 2026-02-05
    - [x] Self-attention blocks (multi-head attention, feedforward, layernorm, dropout). ✅ 2026-02-05
    - [x] Output layer for next-item scores. ✅ 2026-02-05
- [x] Training loop `src/train/train_sasrec.py`: ✅ 2026-02-05
    - [x] Handle training, validation, checkpoint saving. ✅ 2026-02-05
    - [x] Use standard loss (BCE/logistic with sampled negatives). ✅ 2026-02-05
- [x] Implement evaluation metrics: ✅ 2026-02-05
    - [x] HR@K, NDCG@K, MRR@K (K=5 or 10). ✅ 2026-02-05
- [x] Run first baseline experiment: ✅ 2026-02-05
    - [x] Train SASRec to convergence on MovieLens-1M. ✅ 2026-02-05
    - [x] Log metrics and compare roughly to standard ranges from papers/tutorials. ✅ 2026-02-05
- [x] Optional: implement **GRU4Rec** baseline or reuse an existing implementation. ✅ 2026-02-05
- [x] Save baseline configs in `configs/baselines/` and log results in CSV. ✅ 2026-02-05
- [x] Summarize baseline performance in `03_baselines` (table: model × HR@10, NDCG@10, MRR@10). ✅ 2026-02-05

---

## Phase 4 – Item graph & GNN (global item encoder)

Note: Document in `04_model_math`.

- [ ] Define **item co-occurrence graph** construction:
    - [ ] From each user's sequence, slide a window of size $w$ (e.g., 3–5).
    - [ ] For each window, connect all pairs of items with an undirected edge.
    - [ ] Accumulate edge counts over all users.
    - [ ] Optionally drop edges with count below a threshold to keep graph sparse.
- [ ] Implement graph builder `src/data/build_item_graph.py`:
    - [ ] Input: processed sequences.
    - [ ] Output: adjacency list or sparse matrix (for PyG/DGL).
- [ ] Implement **GNN encoder** `src/models/item_gnn.py`:
    - [ ] Choose architecture: simple GCN, GraphSAGE, or LightGCN.
    - [ ] Use 1–2 layers, ReLU, dropout.
    - [ ] Input: initial item embeddings $E^{(0)}$, adjacency graph.
    - [ ] Output: graph-enhanced embeddings $G$ (dimension $d_g$).
- [ ] Decide training mode:
    - [ ] Joint training with SASRec from scratch (simpler) or
    - [ ] Pretrain GNN briefly on some proxy objective then finetune (optional).
- [ ] In `04_model_math`, write formulas for:
    - [ ] Graph construction (edge definition).
    - [ ] GNN layers (e.g., GCN update rule).

---

## Phase 5 – Length-adaptive fusion (novelty)

Note: This is your **main contribution**; document math clearly in `04_model_math`.

### 5.1 Define fusion conceptually

- [ ] Fix notation:
    - [ ] $e_i \in \mathbb{R}^{d_e}$: SASRec item embedding for item $i$.
    - [ ] $g_i \in \mathbb{R}^{d_g}$: GNN item embedding for item $i$.
    - [ ] $L(u)$: history length for user $u$.
    - [ ] $\alpha(u)\in[0,1]$: weight for sequence vs graph for user $u$.
- [ ] Define **length bins**:
    - [ ] Example:
        - Short: $L(u) \leq L_{\text{short}}$
        - Medium: $L_{\text{short}} < L(u) \leq L_{\text{long}}$
        - Long: $L(u) > L_{\text{long}}$.
    - [ ] Choose initial thresholds (e.g., 10 and 50).
- [ ] Define a simple $\alpha(u)$ function:
    - [ ] e.g.,
        - Short: $\alpha(u) = \alpha_{\text{short}}$ (more weight on GNN)
        - Medium: $\alpha(u) = \alpha_{\text{mid}}$
        - Long: $\alpha(u) = \alpha_{\text{long}}$ (more weight on SASRec)
    - [ ] Start with hand-chosen values (e.g. 0.3, 0.5, 0.7) and later allow them to be learned.
- [ ] Define **user-specific fused embedding**:
    - [ ] Ensure dimensions: if $d_e \neq d_g$, add projection to common dimension.
    - [ ] Equation:
        - $g_i' = W_g g_i$ (project GNN embedding if needed).
        - $h_i^{(u)} = \alpha(u), e_i + (1 - \alpha(u)), g_i'$.

### 5.2 Implement fusion in code

- [ ] Implement `compute_alpha` in `src/models/length_adaptive_fusion.py`:
    - [ ] Input: batch of user sequence lengths.
    - [ ] Output: batch of $\alpha(u)$ values (one per user).
- [ ] Add fusion logic into your model:
    - [ ] During batch creation, compute lengths per user.
    - [ ] Broadcast $\alpha(u)$ over all positions in that user's sequence.
    - [ ] For each item id in batch, look up $e_i$ and $g_i'$, then compute $h_i^{(u)}$.
    - [ ] Feed $h_i^{(u)}$ into SASRec instead of plain $e_i$.
- [ ] Ensure compatibility with padding/masks:
    - [ ] $\alpha(u)$ depends only on length, so it's per user, not per position.
    - [ ] Does not interfere with attention masks.

### 5.3 Integrate training

- [ ] Create hybrid model class, e.g. `HybridSASRecGNNLengthAdaptive`:
    - [ ] Contains item embedding table $e_i$.
    - [ ] Contains GNN item embedding module (or table if precomputed).
    - [ ] Computes fused embeddings $h_i^{(u)}$.
    - [ ] Uses SASRec Transformer layers and output head.
- [ ] Implement training script `src/train/train_hybrid_length_adaptive.py`:
    - [ ] Similar to SASRec training, but with GNN and fusion active.
    - [ ] Same loss (BCE / sampled softmax) for next-item prediction.
- [ ] Run overfitting test on a tiny subset to check implementation:
    - [ ] Verify loss goes to near zero.
    - [ ] Check that metrics increase.

---

## Phase 6 – Experiments & ablations

Note: Track in `05_experiments`.

### 6.1 Main comparison

- [ ] Fix core experimental settings:
    - [ ] Learning rate, batch size, max sequence length, GNN depth, etc.
    - [ ] Fix random seeds.
- [ ] Models to compare:
    - [ ] MF/BPR (optional, as simple non-sequential baseline).
    - [ ] **SASRec** (baseline).
    - [ ] SASRec + GNN with **fixed fusion** (e.g., $h_i = e_i + g_i'$) – "non-adaptive hybrid".
    - [ ] **Length-adaptive hybrid** (your model).
- [ ] Run experiments on MovieLens-1M:
    - [ ] For each model, train to convergence.
    - [ ] Log HR@K, NDCG@K, MRR@K.
    - [ ] Save results to `experiments/results_main.csv`.

### 6.2 Ablations & analysis

- [ ] Ablation: no GNN (pure SASRec).
- [ ] Ablation: GNN but **fixed** $\alpha$ (no length adaptation).
- [ ] Ablation: vary $\alpha_{\text{short}}, \alpha_{\text{mid}}, \alpha_{\text{long}}$.
- [ ] User-length analysis:
    - [ ] Split users into short, medium, long buckets by history length.
    - [ ] Compute HR@K, NDCG@K, MRR@K per bucket for each model.
    - [ ] Summarize: where does your model help most? (expect strong gains for short-history users).
- [ ] Store these results in `experiments/results_by_length.csv`.

---

## Phase 7 – Results packaging for your teacher

Note: Use `06_results_for_teacher`.

- [ ] Create clean **tables** (in Markdown or LaTeX) with:
    - [ ] Overall metrics (model × HR@10, NDCG@10, MRR@10).
    - [ ] Metrics by history length (short/medium/long).
- [ ] Write a short **method summary** (1–2 paragraphs):
    - [ ] Task and dataset (MovieLens-1M, next-item).
    - [ ] Baseline SASRec.
    - [ ] GNN item encoder (graph construction + GNN).
    - [ ] Length-adaptive fusion formula.
- [ ] Write a short **results summary** (bullet point style):
    - [ ] "Our length-adaptive hybrid improves HR@10 / NDCG@10 over SASRec and non-adaptive hybrid."
    - [ ] "Gains are especially strong for short-history users."
- [ ] Prepare a zipped **package** for your teacher:
    - [ ] Code (with README and configs).
    - [ ] Processed data description.
    - [ ] CSVs with all metrics.
    - [ ] The method + results notes from Obsidian (export or copy).

---

## Phase 8 – Optional polish & extras

- [ ] Try a second dataset (optional): e.g., another MovieLens variant or an Amazon subset.
- [ ] Try a simpler GNN (LightGCN style) to show robustness.
- [ ] Implement a tiny learned function for $\alpha(u)$ (small MLP) instead of fixed bins and compare.

---

# Mathematical Formulation

Below is a math note you can drop into an Obsidian page like `04_model_math.md`. It gives precise notation and equations for: problem, SASRec, item graph + GNN, length-adaptive fusion, training loss, and metrics.

---

## 1. Problem Definition

- Let $U$ be the set of users, $I$ the set of items (movies).
- For each user $u \in U$, we have an interaction sequence

$$S_u = (i_1, i_2, \dots, i_{T_u}),$$

where $i_t \in I$ is the item at time step $t$, sorted by timestamp.

- We use implicit feedback: an interaction means "user consumed/liked the item", derived from MovieLens ratings.

**Goal (next-item sequential recommendation):**

Given the prefix $S_u^{(t)} = (i_1, \dots, i_t)$, predict the next item $i_{t+1}$. Formally, learn a scoring function

$$f_\Theta(u, S_u^{(t)}, j)$$

that outputs a real-valued preference score for any candidate item $j \in I$, where $\Theta$ are model parameters.

At test time, for each user prefix we rank all candidate items $j \in I$ by scores $f_\Theta(u, S_u^{(t)}, j)$ and evaluate top-$K$ metrics (HR@K, NDCG@K, MRR@K).

---

## 2. SASRec Backbone (Transformer Sequence Encoder)

This follows the SASRec formulation.

### 2.1 Input Representations

- Vocabulary size: $|I|$ items.
- Item embedding matrix:

$$E \in \mathbb{R}^{|I| \times d_e},$$

where $d_e$ is embedding dimension.

- For a user sequence $S_u = (i_1, \dots, i_{T_u})$, we truncate to max length $L$. If $T_u > L$, keep the last $L$ items. If $T_u < L$, pad on the left with a special padding token.

Let the truncated sequence be $(i_1, \dots, i_L)$ (with padding as needed). Define item embeddings:

$$x_t = E_{i_t} \in \mathbb{R}^{d_e}, \quad t=1,\dots,L.$$

We also use positional embeddings:

$$P \in \mathbb{R}^{L \times d_e}, \quad p_t = P_t.$$

The initial input to the Transformer is:

$$H^{(0)}_t = x_t + p_t, \quad t=1,\dots,L.$$

Stacked into a matrix:

$$H^{(0)} \in \mathbb{R}^{L \times d_e}.$$

### 2.2 Self-Attention Block

For layer $\ell = 1,\dots,L_{\text{SAS}}$, we apply a masked self-attention block followed by a point-wise feed-forward network, with residual connections and layer normalization.

Given input $H^{(\ell-1)} \in \mathbb{R}^{L \times d_e}$:

1. **Compute queries, keys, values:**

$$Q = H^{(\ell-1)} W_Q, \quad K = H^{(\ell-1)} W_K, \quad V = H^{(\ell-1)} W_V,$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_e \times d_a}$.

2. **Scaled dot-product attention with causal mask** $M$ (to forbid attending to future positions):

$$A = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_a}} + M \right),$$

$$\tilde{H} = A V.$$

3. **Residual + layer norm:**

$$H' = \text{LayerNorm}\left( H^{(\ell-1)} + \text{Dropout}(\tilde{H}) \right).$$

4. **Feed-forward network:**

$$\hat{H} = \text{FFN}(H') = \sigma(H' W_1 + b_1) W_2 + b_2,$$

where $W_1 \in \mathbb{R}^{d_e \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_e}$, $\sigma$ is a non-linearity (e.g., ReLU).

5. **Residual + layer norm:**

$$H^{(\ell)} = \text{LayerNorm}\left( H' + \text{Dropout}(\hat{H}) \right).$$

After $L_{\text{SAS}}$ layers, we obtain:

$$H^{(L_{\text{SAS}})} \in \mathbb{R}^{L \times d_e}.$$

### 2.3 Sequence Representation and Scoring

For next-item prediction, SASRec typically uses the **last non-padded position** as the sequence representation.

Let $t^*$ be the last real item index (ignoring padding). Sequence representation:

$$s_u = H^{(L_{\text{SAS}})}_{t^*} \in \mathbb{R}^{d_e}.$$

For any candidate item $j \in I$, the score is:

$$f_\Theta(u, S_u, j) = s_u^\top E_j,$$

or using a separate output embedding matrix if desired.

---

## 3. Item Co-occurrence Graph

We build a global **item-item graph** from all user sequences.

### 3.1 Nodes and Edges

- **Nodes:** all items $I$, one node per item.
- **Edges:** connect items that co-occur within a sliding window in any user's sequence.

**Construction:**

- For each user $u$, sequence $S_u = (i_1, \dots, i_{T_u})$, choose a window size $w$ (e.g., $w=3$ or $5$).
- For each position $t$, consider window $W_t = {i_t, i_{t+1}, \dots, i_{\min(t+w-1, T_u)}}$.
- For all unordered pairs $(a,b)$ with $a \neq b, a,b \in W_t$, increment co-occurrence count $c_{ab}$ by 1.

After aggregating over all users, define adjacency weights:

$$w_{ab} = \begin{cases} c_{ab} & \text{if } c_{ab} \geq \tau, \ 0 & \text{otherwise}, \end{cases}$$

where $\tau$ is a minimum co-occurrence threshold to prune very rare edges.

We obtain an undirected weighted graph $G = (I, E)$, with weighted adjacency matrix $W \in \mathbb{R}^{|I| \times |I|}$.

### 3.2 Normalized Adjacency

For a simple GCN, we use the normalized adjacency:

- Add self-loops: $\tilde{W} = W + I$.
- Degree matrix: $\tilde{D}_{ii} = \sum_j \tilde{W}_{ij}$.
- Symmetric normalized adjacency:

$$\hat{A} = \tilde{D}^{-1/2} \tilde{W} \tilde{D}^{-1/2}.$$

---

## 4. GNN Item Encoder

We use a simple GCN with $L_{\text{GNN}}$ layers to get graph-enhanced item representations.

### 4.1 Initial Item Embeddings for GNN

Let

$$E^{(0)} \in \mathbb{R}^{|I| \times d_g}$$

be initial item embeddings for the GNN (can be randomly initialized or tied with SASRec item embeddings).

### 4.2 GCN Layers

For layer $\ell = 1,\dots,L_{\text{GNN}}$:

$$E^{(\ell)} = \sigma\big( \hat{A} E^{(\ell-1)} W^{(\ell)} \big),$$

where:

- $E^{(\ell)} \in \mathbb{R}^{|I| \times d_g}$ is the item representation at layer $\ell$.
- $W^{(\ell)} \in \mathbb{R}^{d_g \times d_g}$ is a trainable weight matrix.
- $\sigma$ is a non-linearity (e.g., ReLU).

After $L_{\text{GNN}}$ layers, we obtain graph-enhanced item representations:

$$G = E^{(L_{\text{GNN}})} \in \mathbb{R}^{|I| \times d_g}.$$

Denote the embedding of item $i$ by:

$$g_i = G_i \in \mathbb{R}^{d_g}.$$

If $d_g \neq d_e$, we add a projection:

$$g_i' = W_g g_i + b_g, \quad W_g \in \mathbb{R}^{d_g \times d_e}, , b_g \in \mathbb{R}^{d_e}.$$

---

## 5. Length-Adaptive Fusion (Novelty)

This is the main novelty: we **adapt the fusion of GNN and SASRec embeddings per user**, based on user history length.

### 5.1 History Length and Bins

For each user $u$, let $L(u)$ be the length of their interaction sequence (after preprocessing).

We define three length bins:

- **Short history:** $L(u) \leq L_{\text{short}}$
- **Medium history:** $L_{\text{short}} < L(u) \leq L_{\text{long}}$
- **Long history:** $L(u) > L_{\text{long}}$

with hyperparameters $L_{\text{short}}$, $L_{\text{long}}$ (e.g., 10 and 50).

We assign weights $\alpha_{\text{short}}, \alpha_{\text{mid}}, \alpha_{\text{long}} \in [0,1]$.

**Intuition:**

- Short: smaller $\alpha$ → more reliance on GNN.
- Long: larger $\alpha$ → more reliance on sequence (SASRec).

Define:

$$\alpha(u) = \begin{cases} \alpha_{\text{short}} & \text{if } L(u) \leq L_{\text{short}}, \ \alpha_{\text{mid}} & \text{if } L_{\text{short}} < L(u) \leq L_{\text{long}}, \ \alpha_{\text{long}} & \text{if } L(u) > L_{\text{long}}. \end{cases}$$

Later, $\alpha_{\text{short}}, \alpha_{\text{mid}}, \alpha_{\text{long}}$ can be learned or tuned.

### 5.2 Fused Item Embedding Per User

For each item $i$, we have:

- SASRec item embedding: $e_i \in \mathbb{R}^{d_e}$ (row of $E$).
- Projected GNN item embedding: $g_i' \in \mathbb{R}^{d_e}$ (row of projected $G$).

For user $u$, define the **user-specific fused embedding**:

$$h_i^{(u)} = \alpha(u) , e_i + (1 - \alpha(u)) , g_i'.$$

This means:

- For short-history users $u$: smaller $\alpha(u)$, more weight on $g_i'$ (GNN).
- For long-history users $u$: larger $\alpha(u)$, more weight on $e_i$ (sequence).

In a mini-batch, for each user $u$ and each position $t$ in their sequence with item $i_t$, we look up $e_{i_t}$ and $g_{i_t}'$, compute $h_{i_t}^{(u)}$, then add positional embedding to form:

$$H^{(0)}_t(u) = h_{i_t}^{(u)} + p_t.$$

This $H^{(0)}(u)$ is then fed into the SASRec Transformer layers exactly as before.

---

## 6. Full Model Forward

Putting it together for a user $u$:

1. **Input:** sequence $S_u = (i_1, \dots, i_L)$ (truncated/padded).
2. **Compute** history length $L(u)$ (actual non-padded length).
3. **Determine** $\alpha(u)$ via bins.
4. **For each item** $i_t$:
    - Get $e_{i_t}$ from SASRec item embedding table $E$.
    - Get $g_{i_t}'$ from projected GNN output.
    - Compute fused embedding $h_{i_t}^{(u)} = \alpha(u) e_{i_t} + (1 - \alpha(u)) g_{i_t}'$.
    - Add positional embedding $p_t$ to get $H^{(0)}_t(u)$.
5. **Run** $H^{(0)}(u)$ through SASRec layers to obtain $H^{(L_{\text{SAS}})}(u)$.
6. **Take** last non-padded position $t^_$ and compute sequence representation $s_u = H^{(L_{\text{SAS}})}_{t^_}(u)$.
7. **For any candidate item** $j$, use either $e_j$ or $h_j^{(u)}$ as item vector for scoring; simplest is:

$$f_\Theta(u, S_u, j) = s_u^\top e_j.$$

---

## 7. Training Loss

We train with a binary cross-entropy loss using positive and sampled negative items at each time step.

For each training instance (user $u$, time $t$), we have:

- Positive item $i_{t+1}$.
- A set of sampled negative items $N_{u,t}$ not interacted by user $u$.

Let scores be:

- $s^+ = f_\Theta(u, S_u^{(t)}, i_{t+1})$.
- For each negative $j \in N_{u,t}$, $s^-_j = f_\Theta(u, S_u^{(t)}, j)$.

We can use point-wise BCE loss:

$$\mathcal{L}_{u,t} = - \log \sigma(s^+) - \sum_{j \in N_{u,t}} \log \sigma(- s^-_j),$$

where $\sigma$ is the sigmoid function.

Total loss (sum over all users and time steps, plus regularization):

$$\mathcal{L}(\Theta) = \sum_{u \in U} \sum_{t} \mathcal{L}_{u,t} + \lambda |\Theta|_2^2.$$

Here, $\Theta$ includes $E$, GNN weights $W^{(\ell)}$, projection $W_g$, SASRec parameters, and possibly $\alpha_{\text{short}}, \alpha_{\text{mid}}, \alpha_{\text{long}}$ if learned.

---

## 8. Evaluation Metrics

For each test instance (user $u$, prefix $S_u^{(t)}$, true next item $i_{t+1}$), we rank the true item among all candidate items, or among a set of candidates including negatives.

Let $\text{rank}(u,t)$ be the rank position (1 = top) of the true item in the predicted ranking.

### 8.1 Hit Rate @ K (HR@K)

$$\text{HR@K} = \frac{1}{N} \sum_{(u,t)} \mathbf{1}{ \text{rank}(u,t) \le K },$$

where $N$ is number of test instances.

### 8.2 NDCG @ K

For each instance:

$$\text{DCG@K}(u,t) = \begin{cases} \frac{1}{\log_2(\text{rank}(u,t) + 1)}, & \text{if } \text{rank}(u,t) \le K, \ 0, & \text{otherwise}, \end{cases}$$

and the ideal DCG is $1$ (since only one relevant item at best position).

So:

$$\text{NDCG@K}(u,t) = \text{DCG@K}(u,t),$$

and overall:

$$\text{NDCG@K} = \frac{1}{N} \sum_{(u,t)} \text{NDCG@K}(u,t).$$

### 8.3 MRR @ K

$$\text{MRR@K} = \frac{1}{N} \sum_{(u,t)} \mathbf{1}{\text{rank}(u,t) \le K} \cdot \frac{1}{\text{rank}(u,t)}.$$

---