# Step 1) Define what “similar design + materials” means (scoring contract)

**Goal**
Make “similar” measurable so you don’t build a model that optimizes the wrong thing.

**Key actions**

* Write a short “similarity contract”:

  * **Design**: silhouette/cut, garment type, sleeve length, neckline, pattern (e.g., color-block), details (buttons, collar).
  * **Material**: fabric type (cotton/denim/knit), texture cues (if visible), plus material mentioned in text.

* Decide the v0 ranking objective:

  * “Top-10 should match garment type + 1–2 design cues + material.”

* Decide weights (even rough is fine for v0):

  * e.g. Design 70% / Material 30%.

**Outcome**

* A 1-page spec that defines what the model must prioritize.

**Success**

* You can take 10 example queries and your team agrees on what would be “correct” top-10 results.

---

# Step 2) Data audit + cleaning (trust your training set)

**Goal**
Ensure every training record is usable and consistent.

**Key actions**

* Verify all images decode, are non-empty, and aren’t corrupted.
* Confirm category3 is all None → drop it for v0.
* Check text quality:

  * Does it often mention fabric? (your example does: “cotton fabric”)
  * Does it mention design attributes (pattern, neckline, sleeve)?
* Remove broken / extreme outliers (tiny images, blank text).

**Outcome**

* Clean dataset + a simple report (# removed, reasons).

**Success**

* Training/eval runs don’t crash; random samples look sane.

---

# Step 3) Normalize labels + fix taxonomy inconsistencies

**Goal**
Avoid “shirts vs shirt” and other label noise confusing both training and evaluation.

**Key actions**

* Create a canonical set for category2 (your list of ~16 types is good).
* Merge obvious duplicates (pluralization, casing).
* Decide what to do with ultra-rare classes (e.g., suiting=22):

  * v0 option: keep but don’t over-weight, or temporarily map to “other”.

**Outcome**

* Stable category set + mapping table.

**Success**

* Counts look right (no duplicate class names), evaluation slices are meaningful.

---

# Step 4) Create leakage-safe splits (very important for retrieval)

**Goal**
Prevent the model from cheating by seeing near-identical items in train and test.

**Key actions**

* Group items by product identity using item_ID structure (often contains a base ID + view info).
* Split by group so all views/variants of the same item fall into only one split.
* Stratify by category2 (and category1 if needed).

**Outcome**

* Train/val/test splits you can trust.

**Success**

* When you inspect test queries, you don’t find the same product image/ID in training.

---

# Step 5) Build structured attribute signals from text (materials + design cues)

**Goal**
Teach the model what you care about: materials and design attributes.

**Key actions**

* Define a small attribute schema (v0-friendly):

  * **Material**: cotton, denim, knit, leather, polyester, etc.
  * **Pattern**: solid, stripe, plaid, color-block, graphic.
  * **Neckline**: crew, v-neck, collar, etc.
  * **Sleeve**: sleeveless, short, long.
  * (Optional) closure/detail: buttons, zipper.

* Extract these attributes from your existing text (rule-based is fine for v0; you’re not building an NLP system here).

* Track missingness:

  * If material is missing often, you may rely more on image cues + category priors.

**Outcome**

* For each item: a lightweight “attribute tag set” you can use for training and evaluation.

**Success**

* Spot-check 100 items: extracted material/pattern/neckline is mostly correct and useful.

---

# Step 6) Establish a baseline retrieval system (so you know you’re improving)

**Goal**
Get a working end-to-end v0 baseline ASAP.

**Key actions**

* Use a strong pretrained multimodal embedding model to create vectors.
* Build a simple retrieval:

  * Encode all catalog items → nearest neighbors → top-10.
* Evaluate quickly using proxies:

  * Category agreement@10 (how often top-10 share category2)
  * Material agreement@10 (from extracted material tags)
  * Qualitative “gallery review” (query + top-10 grid)

**Outcome**

* A baseline that returns top-10 in seconds and gives you reference metrics.

**Success**

* The top-10 often “looks reasonable” even before fine-tuning (you can demo it).

---

# Step 7) Train the v0 core model: dual-encoder multimodal similarity

**Goal**
Make embeddings reflect your domain: fashion design + material relevance.

**Key actions**

* Train a dual-encoder (query encoder and catalog encoder), where the query uses:

  * image + description
* Use a contrastive/metric-learning objective so:

  * True matches are closer; non-matches are farther.
* Define positives and negatives (v0-friendly):

  * Strong positives: same item_ID family (different views) if you have them.
  * Weak positives: same category2 + matching key attributes (material/pattern).
  * Negatives: different category2 or conflicting material/pattern.
* Balance training so tees/blouses don’t dominate.

**Outcome**

* A fine-tuned embedding model specialized for fashion similarity.

**Success**

* Beats baseline on:

  * Recall@10 (or category/material agreement@10)
  * Human gallery looks better (more consistent material + design matches)

---

# Step 8) Add an auxiliary “material head” (cheap improvement, big impact)

**Goal**
Force the model to care about materials, not just silhouette.

**Key actions**

* Add a lightweight secondary objective:

  * Predict material tag from image+text (multi-task learning).
* Even if labels are weak (from text extraction), it helps align the representation.

**Outcome**

* Embeddings that separate “cotton tee” vs “denim jacket” more reliably.

**Success**

* Material agreement@10 improves noticeably without hurting design similarity.

---

# Step 9) Add a re-ranking stage (the best v0 quality boost)

**Goal**
Make top-10 really good by re-checking candidates in detail.

**Key actions**

* Two-stage retrieval:

  * Retriever returns top-100 fast (embedding nearest neighbors).
  * Re-ranker scores those 100 more carefully using both image+text pairs:

    * It can learn finer distinctions: pattern, neckline, fabric mention.
* Output final top-10 from re-ranked list.

**Outcome**

* Higher precision at top ranks (exactly what users feel).

**Success**

* Top-10 looks substantially better even if overall recall changes slightly.
* Fewer “wrong material” results in the first 5.

---

# Step 10) Build a real evaluation set (small but high-quality)

**Goal**
Stop guessing. Measure “similar design + material” directly.

**Key actions**

* Create a curated test set (e.g., 200–500 query items).
* For each query, label:

  * “Good match” vs “OK” vs “Bad” for a candidate set (top-50 from baseline is enough).
* Track metrics:

  * nDCG@10 (quality-weighted ranking)
  * Precision@10 (how many of top-10 are Good/OK)
  * Attribute agreement (material, pattern) as diagnostics

**Outcome**

* A reliable scoreboard for iteration.

**Success**

* You can compare experiments and confidently say which model is better.

---

# Step 11) Package the v0 “protocol” (demo-ready system)

**Goal**
Deliver a v0 that behaves like a product feature.

**Key actions**

* Precompute catalog embeddings and store them in an index.
* Define the API contract:

  * input: image + description
  * output: top-10 item_IDs + similarity score + “why” attributes (material/pattern match)
* Add guardrails:

  * if description missing → fall back to image-only retrieval
  * if confidence low → broaden results or show “closest by category”

**Outcome**

* End-to-end prototype: query → top-10 results.

**Success**

* Live demo works: 10 diverse queries, results are consistently sensible, latency feels acceptable.
