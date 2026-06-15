# PromptCraft-SeqRec — Prompt Strategies Reference

> This document explains all 11 prompt strategies used in the PromptCraft-SeqRec experiment.
> Each strategy is a different way of describing an item to the BGE-M3 embedding model.
> The only thing that changes across experiments is this description — the model and architecture stay identical.

---

## The Core Idea

Every strategy follows this pipeline:

```
Item metadata → [Prompt Strategy] → BGE-M3 → 1024-dim vector → PCA → 256-dim → SASRec
```

The question being answered: **does the way you describe an item change recommendation accuracy?**

---

## Strategy 1 — Title Only (Baseline)

```
"CeraVe Moisturizing Cream"
```

**What it does:** Feeds only the raw item title to the embedding model. No extra information.

**Hypothesis:** The item name alone is enough to produce good embeddings.

**Why it matters:** This is exactly what the original paper does. Every other strategy tries to beat this. If nothing beats it, then prompt engineering doesn't matter for recommendation. If something does beat it, we have a finding.

**Best for:** Items whose names already carry rich semantic meaning (e.g. "CeraVe Moisturizing Cream" already tells you category, brand, and use case).

**Worst for:** Items whose names are opaque (e.g. "Portal 2" tells you nothing about what kind of game it is).

---

## Strategy 2 — Structured Tags

```
"CeraVe Moisturizing Cream | Brand: CeraVe | Category: Face Moisturizer | Price: $14.99"
```

**What it does:** Adds structured metadata (brand, category, price) in a pipe-separated format alongside the title.

**Hypothesis:** Explicit structured attributes help the embedding model separate items that have similar names but different properties.

**Why it matters:** Two products might have similar titles but completely different brands or categories. Structured attributes make those differences explicit and unambiguous to the embedding model.

**Key design choice:** Pipe `|` separator makes each field clearly distinct — the model treats them as separate attributes rather than flowing prose.

---

## Strategy 3 — Rich Prose

```
"A deeply hydrating face moisturizer by CeraVe, formulated with ceramides
and hyaluronic acid for dry sensitive skin."
```

**What it does:** Writes a natural language sentence using all available metadata — title, brand, category, and description.

**Hypothesis:** LLMs were trained on natural language, so natural language descriptions produce richer and more semantically meaningful embeddings than structured formats.

**Why it matters:** BGE-M3 was pretrained on billions of natural language sentences. Feeding it a natural sentence may align better with what it learned during pretraining than a pipe-separated tag list.

**Key design choice:** Uses the item description field directly (truncated to 200 chars) which often contains ingredient lists, use cases, and benefits.

---

## Strategy 4 — User-Centric

```
"Users who like this moisturizer enjoy: hydration, sensitive skin care,
non-comedogenic products, ceramide formulas"
```

**What it does:** Reframes the description entirely around user preferences instead of item properties. Describes the item in terms of what kind of user likes it.

**Hypothesis:** Since recommendation is about matching users to items, describing items in terms of user preferences aligns the embedding space better with the recommendation goal.

**Why it matters:** Two items might look very different as products but attract the same type of user. This framing captures that latent user-item relationship directly in the embedding, rather than leaving the model to infer it.

**Key design choice:** The phrase "Users who like X enjoy:" primes the embedding model to think in terms of taste and preference rather than product attributes.

---

## Strategy 5 — Comparative

```
"CeraVe Moisturizing Cream is similar to: Cetaphil Moisturizing Cream,
Vanicream Moisturizer. Appeals to fans of: gentle skincare,
dermatologist-recommended products"
```

**What it does:** Describes items relative to other items by referencing what they are similar to (using the "also bought" field in the metadata).

**Hypothesis:** If the embedding model knows what an item is similar to, it can position it more accurately in embedding space relative to other items, creating better clustering of related products.

**Why it matters:** Recommendation is fundamentally about relative similarity — "if you liked X you'll like Y." Encoding that relative positioning directly into the prompt may help the embedding model build a better similarity structure.

**Key design choice:** Uses the real `also_bought` field from Amazon metadata to get genuine co-purchase relationships rather than invented examples.

---

## Strategy 6 — Hybrid (Original Best Guess)

```
"CeraVe Moisturizing Cream | Category: Face Moisturizer |
For fans of: hydration, ceramides, sensitive skin care"
```

**What it does:** Combines structured attributes (from Strategy 2) with user-centric framing (from Strategy 4) in a single description.

**Hypothesis:** No single information type is optimal — combining item attributes with user preference signals gives the embedding model the most complete picture.

**Why it matters:** This is the authors' hypothesis about what the best single strategy should look like before running any experiments. It bridges the gap between what the item IS (structured) and who it is FOR (user-centric).

**Key design choice:** Deliberately short and focused — only the three most informative fields (title, category, user tags) without cluttering with price, rank, or description.

---

## Strategy 7 — Structured + Comparative

```
"CeraVe Moisturizing Cream | Category: Face Moisturizer | Brand: CeraVe |
Similar to: Cetaphil Moisturizing Cream, Vanicream Moisturizer"
```

**What it does:** Extends Strategy 2 (structured tags) by adding a "Similar to" field with real co-purchased items from the metadata.

**Hypothesis:** Structured attributes + relational context together give the embedding model both what the item is and where it sits relative to other items.

**Why it matters:** Strategy 2 tells the model the item's properties. Strategy 5 tells the model its neighborhood. Combining both gives a richer picture than either alone.

**Key design choice:** Keeps the clean pipe-separated format of Strategy 2 so the relational field is parsed as just another structured attribute, not as flowing prose.

---

## Strategy 8 — Enriched Descriptive

```
"CeraVe Moisturizing Cream by CeraVe in Face Moisturizer - #142 in Beauty.
Features: deeply hydrating, ceramides, hyaluronic acid, dry skin"
```

**What it does:** Uses all available metadata fields in structured prose — title, brand, category, sales rank, and extracted product features from the description.

**Hypothesis:** The more metadata fields included, the more discriminative the embedding. Sales rank captures popularity signal that no other strategy uses.

**Why it matters:** Sales rank is a unique signal — a #1 bestseller and a niche product serve different user needs even if their descriptions are similar. Features extracted from the description capture functional properties that the title alone misses.

**Key design choice:** Includes sales rank (`#142 in Beauty`) which no other strategy uses — tests whether popularity context improves embedding quality.

---

## Strategy 9 — Semantic Context

```
"CeraVe Moisturizing Cream. Category: Face Moisturizer. Made by CeraVe.
Best suited for: deeply hydrating formula with ceramides and hyaluronic acid
for dry sensitive skin"
```

**What it does:** Frames the item in its usage context using natural sentences. Emphasizes "Best suited for:" to highlight the use case rather than just the product attributes.

**Hypothesis:** Framing around usage context (what situation this product solves) produces embeddings that capture functional similarity better than attribute-based descriptions.

**Why it matters:** Two products with completely different names and brands might serve the exact same use case. "Best suited for: dry sensitive skin" captures that functional equivalence in a way that title or brand never could.

**Key design choice:** "Best suited for:" is a deliberate semantic frame that primes the embedding model to think about utility and use case rather than product identity.

---

## Strategy 10 — Multi-View Fusion

```
"CeraVe Moisturizing Cream brand=CeraVe category=Face Moisturizer
price=$14.99 popularity=#142 in Beauty features=(deeply hydrating, ceramides)
similar_to=(Cetaphil Moisturizing Cream, Vanicream Moisturizer)"
```

**What it does:** Packs every available metadata field into a single compact key=value format — title, brand, category, price, popularity, features, and similar items all in one description.

**Hypothesis:** Combining all views (structured, descriptive, relational, popularity) in one prompt gives the embedding model the maximum possible information about the item.

**Why it matters:** Each field captures a different dimension of the item. Price separates budget vs premium products. Popularity separates mainstream vs niche. Similar items give relational context. No other strategy uses all of these simultaneously.

**Key design choice:** The `key=value` format is deliberately compact — it fits the most information into the fewest tokens, staying within BGE-M3's 128-token limit.

---

## Strategy 11 — Review Augmented

```
"CeraVe Moisturizing Cream by CeraVe in Face Moisturizer.
Users say: sensitive skin, great moisturizer, long lasting,
non greasy, dry skin, absorbs quickly"
```

**What it does:** Augments the title + structured info with real phrases extracted from user reviews using bigram/trigram frequency analysis.

**Hypothesis:** Real user language captured from reviews contains nuanced signals about how people actually experience and talk about products — signals that structured metadata completely misses.

**Why it matters:** Metadata is written by the seller. Reviews are written by actual users. Review phrases like "absorbs quickly", "non greasy", "great for winter" reflect real usage experiences that no product description captures. These phrases may align better with how users think about items when making purchase decisions.

**Key design choice:** Uses bigram and trigram extraction (2 and 3-word phrases) rather than single words to capture meaningful expressions. Filtered by minimum frequency (≥2 mentions) to remove noise. Top 8 phrases per item.

---

## Summary Table

| #   | Strategy                 | Extra Info Used                                | Key Hypothesis                                   |
| --- | ------------------------ | ---------------------------------------------- | ------------------------------------------------ |
| T1  | Title Only               | None                                           | Name alone is enough                             |
| T2  | Structured Tags          | Brand, Category, Price                         | Structured attributes improve separation         |
| T3  | Rich Prose               | Brand, Category, Description                   | Natural language aligns with LLM pretraining     |
| T4  | User-Centric             | Category, Tags                                 | User preference framing aligns with rec goal     |
| T5  | Comparative              | Similar items, Tags                            | Relative positioning improves embedding space    |
| T6  | Hybrid                   | Category, Brand, Tags                          | Combining item + user info is optimal            |
| T7  | Structured + Comparative | Brand, Category, Similar items                 | Structure + relations together beat either alone |
| T8  | Enriched Descriptive     | All fields + Sales Rank + Features             | More metadata = more discriminative              |
| T9  | Semantic Context         | Category, Brand, Description (use-case framed) | Usage context captures functional similarity     |
| T10 | Multi-View Fusion        | All fields in key=value format                 | Maximum information = best embeddings            |
| T11 | Review Augmented         | Brand, Category + real user review phrases     | User language captures signals metadata misses   |

---

## Expected Pattern of Results

| Dataset           | Expected prompt sensitivity | Reason                                                                 |
| ----------------- | --------------------------- | ---------------------------------------------------------------------- |
| **Steam Gaming**  | HIGH                        | Game titles like "Portal 2" carry no semantic info — tags fill the gap |
| **MovieLens**     | MEDIUM                      | Titles partially informative — genre/year adds value                   |
| **Amazon Beauty** | LOW                         | "CeraVe Moisturizing Cream" already encodes category and brand         |

**Winning condition:** Any strategy beats T1 (title only) on NDCG@10 on at least 2 out of 3 datasets.

---

_PromptCraft-SeqRec — Research Reference Document_
