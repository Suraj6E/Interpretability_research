# Planning: Interpretability Research

This is a living document. Items marked **(H)** are hypotheses we plan to test, not established facts.

---

## 1. Vision

Build an interactive, visual map of an LLM's insides: for any question, show which parts fired, which pushed the answer, and which were necessary. Start small (GPT-2 small) and build up to larger models and finer concepts.

## 2. Goals and non-goals

**Goals**
- **G1:** A visual model map for every question in the question set.
- **G2:** Check our methods against circuits already published for GPT-2 small.
- **G3:** Compare skills: which components are shared and which are specific.
- **G4:** Beginner-friendly and reproducible on free Google Colab.
- **G5:** A foundation to build up from (sparse-autoencoder features, larger models, attribution graphs).

**Non-goals (for now)**
- Analyzing closed models (ChatGPT, Claude). Their internals aren't accessible.
- Claiming a full explanation of any model.
- "Activating every part" of the model. Neuron-level coverage is easy to reach and meaningless, and the number of concept-level features has no fixed size.
- Training our own models or running a benchmark leaderboard.

## 3. Research questions

| ID | Question | Hypothesis |
|---|---|---|
| RQ1 | Which components are necessary for each skill? | **(H1)** Each skill depends on a small fraction of the 144 heads and 12 MLPs. |
| RQ2 | Which components are shared across skills? | **(H2)** Some early-layer and late-layer components are shared; middle-layer components are more skill-specific. |
| RQ3 | Does activation strength predict causal importance? | **(H3)** Only weakly. Many strongly active parts won't be necessary. |
| RQ4 (later) | On two-step questions, does a larger model reason step by step or use a memorized shortcut? | **(H4)** It depends on the question. Tested on Gemma-2-2b. |

RQ3 is especially useful because it directly tests the project's central caveat that "lit up" is not the same as "used."

## 4. Definitions (metrics)

| Metric | Definition | Meaning |
|---|---|---|
| **Logit difference (LD)** | logit(correct answer) − logit(wrong answer) | How strongly the model prefers the right answer |
| **Activation strength** | Size (L2 norm) of a head's or MLP's output at the final token | How strongly a part fired |
| **Direct logit attribution (DLA)** | A part's output projected onto the "correct minus wrong" answer direction, after the final layer norm | How much a part directly pushed toward the right answer |
| **Patching recovery** | (LD_patched − LD_twin) / (LD_clean − LD_twin) | 0 = no effect, 1 = this part alone restores the full answer |

## 5. Question set design

**Rules**
1. The answer must be a **single token**, so the logit difference is simple to compute.
2. A question and its twin differ in **one detail only**.
3. A question and its twin have the **same number of tokens**, so positions line up for patching.
4. Keep only questions the model answers correctly (LD > 0). A wrong answer has no mechanism to explain.
5. Aim for 30–50 questions per skill, generated from templates.

**Format (one entry)**

```json
{
  "id": "ioi_001",
  "skill": "indirect_object",
  "clean": "When Mary and John went to the store, John gave a drink to",
  "twin": "When Mary and John went to the store, Mary gave a drink to",
  "correct": " Mary",
  "wrong": " John",
  "reference": "Wang et al. 2022"
}
```

**Skills (version 1):** induction, indirect object, greater-than, fact recall, grammar agreement. Two-step reasoning is added in Phase 5, since it needs a larger model.

## 6. Architecture

```
questions/*.json
      │
      ▼
interp/run.py      load model (TransformerLens), run clean + twin, cache activations
      │
      ▼
interp/metrics.py  activation strength, DLA, patching recovery
      │
      ▼
results/<model>/<question_id>.json
      │
      ▼
viewer/index.html  static HTML + JavaScript, reads JSON, no server needed
                   (can be hosted free on GitHub Pages)
```

**Result format (one question)**

```json
{
  "id": "ioi_001",
  "model": "gpt2-small",
  "logit_diff": { "clean": 3.2, "twin": -3.1 },
  "activation": { "heads": [[0.0]], "mlp": [0.0] },
  "dla":        { "heads": [[0.0]], "mlp": [0.0] },
  "patching":   { "heads": [[0.0]], "mlp": [0.0] },
  "attention":  { "tokens": ["When", " Mary"], "patterns": "stored separately" }
}
```

(`heads` is a layers × heads grid, 12 × 12 for GPT-2 small. `mlp` has one value per layer.)

## 7. Visualizations

| ID | View | Design |
|---|---|---|
| V1 | **Model map** | Grid: rows = layers 0–11, columns = heads 0–11 plus one MLP column. Toggle between activation, DLA and patching. |
| V2 | **Clean vs. twin diff** | Same grid, colored by the difference between the two runs |
| V3 | **Attention viewer** | Click a head to see which earlier tokens it attends to |
| V4 | **Skill fingerprints** | Small grids side by side, one per skill |
| V5 | **Activation vs. causal scatter** | Each dot is one component: x = activation strength, y = patching recovery (answers RQ3) |

**Design rules**
- Use a colorblind-safe palette. DLA and patching use a diverging scale (negative / zero / positive); activation uses a single-hue scale.
- Every cell shows its exact value and component name on hover.
- No server and no build step: one HTML page that opens directly in a browser.
- Every view has a short "how to read this" note beside it.

## 8. Phases and milestones

Steps are listed in order. Timing depends on available time, so no dates are set.

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Setup** | README, PLANNING, LICENSE, question set v0 | Repository is public and the question set is drafted |
| **1. Activation map** | Recording script and V1 (activation only) | Maps render for all version-1 skills |
| **2. Contribution map** | DLA and V3 attention viewer | DLA shown in V1; attention viewer works |
| **3. Causal map** | Activation patching, V2, validation | Known results reproduced qualitatively (e.g., induction heads appear in layers 5–7; IOI name-mover heads appear in late layers) |
| **4. Comparison** | V4, V5, written summary of RQ1–RQ3 | Findings written up, including what did *not* work |
| **5. Build up** | Gemma-2-2b, sparse-autoencoder features (SAELens), attribution graphs (circuit-tracer), two-step study (RQ4) | First concept-level map published |

## 9. Validation plan (not fooling ourselves)

1. **Known-circuit check:** reproduce published GPT-2 results before trusting new findings.
2. **Random baseline:** patch or ablate random heads as a control.
3. **Ablation choice:** compare zero ablation with mean ablation and report both if they disagree.
4. **Many prompts:** report averages and spread over the whole question set, not the best example.
5. **Backup mechanisms:** remember that removing one part can let others take over, so a small effect doesn't always mean "unimportant."
6. **Report failures:** negative results go into the write-up too.

## 10. Risks and honest difficulties

| Risk | Impact | Mitigation |
|---|---|---|
| Activation maps get read as "importance" | Misleading conclusions | All three views side by side, with a caveat on every view |
| GPT-2 findings don't carry over to bigger models | Limited generality | State this clearly; Phase 5 tests a larger model |
| Twin questions differ in token length | Patching positions don't line up | Check token lengths automatically and drop mismatches |
| GPT-2 small can't do some skills | Nothing to explain | Filter to questions with LD > 0; leave hard skills to Phase 5 |
| Neurons are polysemantic | Hard to label parts | Treat labels as hypotheses; add sparse-autoencoder features later |

## 11. Community and outreach

- Publish the viewer on GitHub Pages so anyone can try it without installing anything.
- Add a 1280×640 social preview image showing a real model map.
- Open "good first issue" tasks (new question sets, new visualizations).
- Write a short post for each milestone, with screenshots and honest findings.

## 12. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-12 | Start with GPT-2 small | Many published circuits to check against; runs on free Colab |
| 2026-10-12 | Use TransformerLens | Standard library with easy access to every activation |
| 2026-10-12 | Static HTML viewer | No server; free hosting on GitHub Pages |
| 2026-10-12 | Three views (activation, contribution, causal) | "Activated" alone is misleading |
| 2026-10-12 | License: MIT (proposed) | Simple and permissive; encourages reuse |

## 13. Open questions

- Colab notebook first, or a local Python package first?
- Should Phase 5 use Gemma-2-2b or another open model with pretrained sparse autoencoders?
- Keep the repository name `Interpretability_research`, or switch to hyphens (see the README notes)?