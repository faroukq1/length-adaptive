Congratulations Farouk on completing your experiments with results on both ML-100K and ML-1M beating BERT4Rec! Here is a precise, step-by-step guide to write a publishable IEEE conference paper from your work.

---

## Step 1 – Choose the Right Venue First

Before writing a single word, target a specific IEEE conference. Your model fits best in:

- **IEEE ICDM** (International Conference on Data Mining) – CORE A\*, deadline ~June 2026, directly covers sequential recommendation [recsys](https://recsys.info)
- **IEEE ICDM / IEEE BigData** – applied ML + recommender systems tracks
- **IEEE ICTAI** or **IEEE SSCI** – more accessible, good for Master's-level work, broader acceptance

Why this matters: **paper length, formatting, and emphasis differ by venue**. IEEE conferences typically require 6–8 pages in double-column, 10pt font using the official IEEE template. Download the LaTeX/Word template at `ieee.org/conferences/publishing/templates`. [ieee](https://www.ieee.org/conferences/publishing/templates)

---

## Step 2 – Write the Title & Abstract Last (But Plan Now)

Draft a working title now like:

> _"Length-Adaptive Hybrid GNN-Transformer for Sequential Recommendation"_

The abstract must be a single paragraph ≤250 words, self-contained (no citations, no undefined acronyms), and must state: problem → gap → method → key result. Write or finalize this **after** completing all other sections. [scribd](https://www.scribd.com/document/943352194/IEEE-Conference-Paper-Format-and-Structure)

---

## Step 3 – Write the Introduction (Section I)

This is the most critical section for reviewers. Structure it as four clear paragraphs:

1. **Hook** – Sequential recommendation is important; cite scale and real-world impact
2. **Problem & gap** – Transformers (SASRec, BERT4Rec) ignore global item structure; existing GNN+Transformer hybrids use fixed fusion regardless of user history length
3. **Your solution** – Propose length-adaptive fusion: short-history users lean on the GNN graph signal; long-history users lean on the Transformer sequence signal
4. **Contributions** – List 3 bullet points:
   - Novel length-adaptive fusion mechanism (the core novelty)
   - Empirical evaluation on MovieLens-100K and MovieLens-1M outperforming BERT4Rec
   - Analysis by user history length showing where gains are strongest

IEEE reviewers check that contributions are **explicitly stated and verifiable** against your results. [ieee-pes](https://ieee-pes.org/publications/authors-kit/conference-organizer-and-reviewer-guidelines/)

---

## Step 4 – Write the Related Work Section (Section II)

Use a **funnel structure**: broad → specific. Cover three subsections: [gist.github](https://gist.github.com/ikbelkirasan/848f97c4a1aee1fa6277ced7b5be80af)

- **Sequential recommendation models**: cite GRU4Rec, SASRec (Kang & McAuley 2018), BERT4Rec (Sun et al. 2019)
- **GNN-based recommendation**: cite LightGCN, SR-GNN, SURGE (graph + sequence hybrid)
- **Hybrid models & fusion**: cite works that combine GNN and Transformer, then show none of them do _length-adaptive_ fusion

**Critical framing**: note the BERT4Rec vs SASRec debate — a RecSys 2023 paper showed SASRec can outperform BERT4Rec when loss functions are controlled equally. This strengthens your baseline choice and shows you understand the literature rigorously. [arxiv](https://arxiv.org/abs/2309.07602)

Each paragraph should end with a sentence explaining **why that line of work is insufficient** and pointing toward your gap.

---

## Step 5 – Write the Methodology Section (Section III)

This is your longest section. Use your existing math notes directly. Structure as:

1. **Problem formulation** – formal definition of next-item prediction (your Section 1 math)
2. **SASRec backbone** – embedding, self-attention blocks, scoring (your Section 2 math)
3. **Item co-occurrence graph & GNN encoder** – graph construction with window \(w\), edge weights, GCN update rule \(E^{(\ell)} = \sigma(\hat{A} E^{(\ell-1)} W^{(\ell)})\) (your Section 4 math)
4. **Length-adaptive fusion** – bins \(L*\text{short}, L*\text{long}\), the \(\alpha(u)\) function, fused embedding \(h_i^{(u)} = \alpha(u)\, e_i + (1-\alpha(u))\, g_i'\) (your Section 5 math — **this is your novelty, give it the most space**)
5. **Training objective** – BCE loss with negative sampling (your Section 7 math)

Include a **model architecture figure** (a diagram showing the pipeline: sequence → fused embeddings → Transformer → scoring). IEEE papers are visual; a clear figure is worth more than two paragraphs. [scribd](https://www.scribd.com/document/943352194/IEEE-Conference-Paper-Format-and-Structure)

---

## Step 6 – Write the Experiments Section (Section IV)

Organize as sub-sections:

- **Datasets**: describe ML-100K and ML-1M (stats: users, items, interactions, density, split strategy)
- **Baselines**: list GRU4Rec, SASRec, BERT4Rec, and your two ablations (fixed-fusion hybrid, your length-adaptive hybrid)
- **Implementation details**: hyperparameters, learning rate, batch size, max sequence length, GNN depth, seeds — **always report seeds for reproducibility** [ieee-pes](https://ieee-pes.org/publications/authors-kit/conference-organizer-and-reviewer-guidelines/)
- **Main results table**: models × HR@10, NDCG@10, MRR@10 on both datasets. Bold the best number in each column
- **Per-length-bucket analysis**: HR@10 and NDCG@10 split by short/medium/long history — **this is your killer result**, show the model helps most for short-history users
- **Ablation study**: rows = pure SASRec / GNN fixed fusion / length-adaptive. This isolates your contribution cleanly

---

## Step 7 – Write the Conclusion (Section V)

Keep it to one paragraph. State: what you proposed, what was validated, and one or two **future directions** (e.g., learned \(\alpha\) via MLP, extension to other domains). Do not repeat numbers here — reviewers already read your results. [scribd](https://www.scribd.com/document/943352194/IEEE-Conference-Paper-Format-and-Structure)

---

## Step 8 – References (IEEE Style)

Use numbered citations in square brackets in order of appearance, e.g. ` [recsys](https://recsys.info)`, ` [ieee](https://www.ieee.org/conferences/publishing/templates)`. Format each reference as: [sourcely](https://www.sourcely.net/post/ieee-citation-format-a-complete-guide-for-engineering-and-computer-science-students)

> [n] A. Author, "Title," in _Proc. Conference Name_, Year, pp. X–Y.

Aim for **15–25 references**. Include the SASRec, BERT4Rec, LightGCN, SR-GNN original papers, and the MovieLens dataset paper. Use a reference manager (Zotero, JabRef) with the IEEE BibTeX style.

---

## Step 9 – Final Checks Before Submission

Run through this checklist:

- [ ] Paper fits within page limit (6–8 pages depending on venue) [ieeeindicon](https://ieeeindicon.org/guidelines-and-policies/)
- [ ] Two-column IEEE template used, 10pt font [ieeeindicon](https://ieeeindicon.org/guidelines-and-policies/)
- [ ] Abstract is ≤250 words, no citations inside it [scribd](https://www.scribd.com/document/943352194/IEEE-Conference-Paper-Format-and-Structure)
- [ ] Contributions are explicitly listed in the introduction [ieee-pes](https://ieee-pes.org/publications/authors-kit/conference-organizer-and-reviewer-guidelines/)
- [ ] All tables have bolded best results, all figures have captions
- [ ] Novelty is clearly differentiated from baselines in related work
- [ ] No author names or affiliations in blind submission versions [ieeeindicon](https://ieeeindicon.org/guidelines-and-policies/)
- [ ] Plagiarism check via iThenticate (IEEE requires this) [ieeevis](https://ieeevis.org/year/2025/info/call-participation/paper-submission-guidelines)
- [ ] Proofread for English grammar (Grammarly or a native speaker pass)

---

## Writing Order (Recommended)

| Order | Section          | Why                                                                                                                               |
| ----- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1st   | Methodology      | You know this best — write while it's fresh                                                                                       |
| 2nd   | Experiments      | Results are already done                                                                                                          |
| 3rd   | Related Work     | Positions your work after you know what you built                                                                                 |
| 4th   | Introduction     | Write once you know the full story                                                                                                |
| 5th   | Conclusion       | Summarize what you proved                                                                                                         |
| Last  | Abstract & Title | Polish once everything is complete [scribd](https://www.scribd.com/document/943352194/IEEE-Conference-Paper-Format-and-Structure) |

---

## One Strategic Note on BERT4Rec as Baseline

Since you beat BERT4Rec, be careful about the **loss function fairness issue** — reviewers familiar with the RecSys 2023 paper may question whether BERT4Rec was trained with its native cross-entropy-over-all-items loss. Explicitly state in your experiments section which loss each baseline uses. This shows methodological rigor and preempts reviewer criticism. [alphaxiv](https://www.alphaxiv.org/overview/2309.07602v1)
